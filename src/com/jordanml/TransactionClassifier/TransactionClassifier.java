package com.jordanml.TransactionClassifier;

import java.io.File;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collection;
import java.util.Properties;

// RSESLIB
import rseslib.processing.reducts.AllGlobalReductsProvider;
import rseslib.structure.table.ArrayListDoubleDataTable;
import rseslib.structure.table.DoubleDataTable;
// For reporting progress to console
import rseslib.system.Report;
import rseslib.system.output.StandardErrorOutput;
import rseslib.system.output.StandardOutput;
import rseslib.system.progress.StdOutProgress;

// WEKA
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class TransactionClassifier
{
    // The class index to be considered positive when classifying instances
    public static final int POSITIVE_CLASS_INDEX = 1;
    
    public static void main(String[] args)
    {
        long start = System.nanoTime(); // Time that execution started
        checkArgs(args);
        long end = System.nanoTime(); // Time that execution ended
        System.out.println("Terminating: " + (end - start)/1000000);
    }
    
    /**
     * Checks the provided command-line arguments and proceeds appropriately
     * 
     * @param args The provided command-line arguments
     */
    public static void checkArgs(String[] args)
    {
        if(args.length < 1)
        {
            System.out.println("No arguments provided");
            printProperUsage();
        }
        else
        {
            // Check what mode the first argument specifies
            switch(args[0].toLowerCase())
            {
                case "split":
                    // Check for proper arguments (a single file and number of folds)
                    trySplit(args);
                    break;
                case "test-once":
                    testOnce(args);
                    break;
                case "multi":
                    tryMultithread(args);
                    break;
                case "help":
                    printProperUsage();
                    break;
                default:
                    System.out.println("Invalid mode provided");
                    printProperUsage();
                    break;
            }
        }
    }
    
    /**
     * Splits a dataset into the specified number of folds and saves each fold as a separate .arff file.
     * 
     * @param args The command-line arguments
     */
    public static void trySplit(String[] args)
    {
        String filepath, savePath;
        int numFolds;
        Dataset dataset;
        
        // Check for proper number of arguments
        if(args.length < 4)
        {
            System.out.println("Not enough arguments for mode 'split'");
        }
        else
        {
            filepath = args[1];
            savePath = args[3];
            
            try
            {
                // Get integer from command-line argument
                // An exception will be thrown if the argument is not an integer
                numFolds = Integer.parseInt(args[2]);
            }
            catch(NumberFormatException e)
            {
                System.out.println("Expected an integer for number of folds");
                printProperUsage();
                return;
            }
            
            // Load the data
            dataset = new Dataset(filepath);
            
            if(!dataset.hasData())
            {
                System.out.println("No dataset loaded");
                return;
            }
            else
            {
                // Split the data
                DatasetSplitter splitter = null;
                
                splitData(splitter, dataset, numFolds, savePath, true);
            }
        }
    }
    
    /**
     * Checks provided arguments and attempts to run multi-threaded cross validation
     * @param args
     */
    public static void tryMultithread(String args[])
    {
        String datasetPath = null;
        String resultsPath = null;
        String savePath = null;
        float learningRate, momentum;
        int numFolds, reductMode;
        
        if(args.length < 8)
        {
            System.out.println("Not enough arguments for mode: multi");
            return;
        }
        else
        {
            datasetPath = args[1];
            resultsPath = args[2];
            savePath = args[3];
            
            try
            {
                numFolds = Integer.parseInt(args[4]);
                learningRate = Float.parseFloat(args[5]);
                momentum = Float.parseFloat(args[6]);
                reductMode = Integer.parseInt(args[7]);
            }
            catch(NumberFormatException e)
            {
                System.out.println("Invalid non-numeric argument.");
                printProperUsage();
                return;
            }
            
            if(reductMode != 1 && reductMode != 2)
            {
                System.out.println("Invalid reduct mode. Must be 1 or 2.");
                printProperUsage();
                return;
            }
            
            Dataset dataset = new Dataset(datasetPath);
            
            if(!dataset.hasData())
            {
                System.out.println("Failed to load data from " + datasetPath);
                return;
            }
            
            multithreadCV(resultsPath, dataset, numFolds, savePath, learningRate, momentum, reductMode);
            
        }
    }
    /**
     * Checks arguments for the testOnce mode. If arguments are valid, calls testOnceClassify
     * and trains and evaluates a model with the given training set, evaluation set, learning rate,
     * momentum, and reduct mode.
     * 
     * @param args
     */
    public static void testOnce(String[] args)
    {
        String filepathTrain, filepathTest, resultsPath;
        Dataset trainingSet, testingSet;
        float learningRate, momentum;
        int reductMode;
        
        if(args.length < 7)
        {
            System.out.println("Not enough arguments for mode 'test-once'");
            printProperUsage();
        }
        else
        {
            filepathTrain = args[1];
            filepathTest = args[2];
            resultsPath = args[3];
            
            try
            {
                learningRate = Float.parseFloat(args[4]);
                momentum = Float.parseFloat(args[5]);
                reductMode = Integer.parseInt(args[6]);
            }
            catch(NumberFormatException e)
            {
                System.out.println("Learning rate and momentum are expected as floats between 0.0 and 1.0");
                printProperUsage();
                return;
            }
            
            if(learningRate < 0.0 || learningRate > 1.0 || momentum < 0.0 || momentum > 1.0)
            {
                System.out.println("Learning rate and momentum are expected as floats between 0.0 and 1.0");
                printProperUsage();
                return;
            }
            
            if(reductMode != 1 && reductMode != 2)
            {
                System.out.println("Reduct-mode can only be set to 1 (M-All) or 2 (M-Dec)");
                printProperUsage();
                return;
            }
            
            // Load the training set
            trainingSet = new Dataset(filepathTrain);
            trainingSet.setClassIndex(trainingSet.numAttributes() - 1);
            // Load the testing set
            testingSet = new Dataset(filepathTest);
            testingSet.setClassIndex(testingSet.numAttributes() - 1);
            
            if(!testingSet.hasData())
            {
                System.out.println("Could not load testing set");
            }
            else if(!trainingSet.hasData())
            {
                System.out.println("Could not load training set");
            }
            else
            {
                Evaluation results = testOnceClassify(trainingSet, testingSet, learningRate, momentum, reductMode);
                saveResults(results, resultsPath);
            }
        }
    }

    /**
     * Saves the results produced by evaluation of a classifier.
     * 
     * @param results
     * @param path path to save results to 
     */
    public static void saveResults(Evaluation results, String path)
    {
        if(results == null)
        {
            System.out.println("Empty results -> Could not run classifier");
        }
        else
        {
            try
            {

                PrintWriter printResults = new PrintWriter(new FileWriter(path));
                printResults.printf("TP: %f%nFP: %f%nTN: %f%nFN: %f\n", results.numTruePositives(POSITIVE_CLASS_INDEX),
                                                                      results.numFalsePositives(POSITIVE_CLASS_INDEX),
                                                                      results.numTrueNegatives(POSITIVE_CLASS_INDEX),
                                                                      results.numFalseNegatives(POSITIVE_CLASS_INDEX));
                printResults.close();
                System.out.println("Saved results to " + path);
            }
            catch(Exception e)
            {
                System.out.println("Could not save results");
            }
        }
    }
    
    /**
     * Runs multi-threaded cross validation. Each run of cross validation is executed in a separate thread.
     * 
     * @param resultsPath base path to save results to 
     * @param dataset the source dataset
     * @param numFolds the number of folds
     * @param savePath path to save folds .arff files to 
     * @param learningRate the learning rate for backpropagation
     * @param momentum the momentum for backpropagation
     */
    public static void multithreadCV(String resultsPath, Dataset dataset, int numFolds, String savePath, float learningRate, float momentum, int reductMode)
    {
        // Initialize start time
        long t_start = System.nanoTime();
        
        CrossValidationThread threads[] = new CrossValidationThread[numFolds];
        
        
        // split dataset into multiple folds
        DatasetSplitter splitter = null;
        String foldPaths[] = splitData(splitter, dataset, numFolds, savePath, false);
        int firstIndex[] = new int[numFolds];
        
        // Combine the folds into a single dataset
        ArrayList<Dataset>folds = new ArrayList<Dataset>();
        
        for(int i = 0; i < numFolds; i++)
        {
            if(i == 0)
                firstIndex[i] = 0;
            else
                firstIndex[i] = firstIndex[i - 1] + folds.get(i - 1).numInstances() - 1;
            
            folds.add(new Dataset(foldPaths[i]));
        }
        
        // Combine folds again
        Dataset fullSet = makeTrainingSet(folds);
        
        for(int i = 0; i < numFolds; i++)
        {
            // get each dataset and test set
            Dataset test = getTestDataset(fullSet, firstIndex[i], folds.get(i).numInstances());
            Dataset train = getTrainDataset(fullSet, firstIndex[i], folds.get(i).numInstances());
            test.setClassIndex(test.numAttributes() - 1);
            train.setClassIndex(train.numAttributes() - 1);
            test.setName(savePath + "fold_" + i + "_test");
            train.setName(savePath + "fold_" + i + "_train");
            // run testOnceClassify
            threads[i] = new CrossValidationThread();
            threads[i].init(resultsPath + "_fold_" + i, train, test, i, learningRate, momentum, reductMode);
            threads[i].start();
        }
        folds = null;
        
        for(int i = 0; i < numFolds; i++)
        {
            try
            {
                threads[i].join(); 
            }
            catch(InterruptedException e)
            {
                System.err.println("Thread handling fold " + i + " was interrupted");
                return;
            }
        }
        
        long t_end = System.nanoTime();
        
        System.out.println("All threads have completed their jobs. Time: " + (t_end - t_start)/1000000 + " ms");
    }
    
    /**
     * Makes a new dataset by copying sequential instances from a given source
     * @param source the source dataset
     * @param first the first instance to be copied
     * @param numInstances the number of instances to be copied
     * @return returns the new dataset
     */
    public static Dataset getTestDataset(Dataset source, int first, int numInstances)
    {
        Instances test = new Instances(source.getInstances(), numInstances);
        
        for(int i = 0; i < numInstances; i++)
        {
            test.add(source.getInstances().get(first + i));
        }
        
        return new Dataset(test);
    }
    
    /**
     * Makes a new dataset by adding all instances from a given source
     * EXCEPT for instances in the range [testFirst, testFirst + testInstances - 1]
     * @param source the source dataset
     * @param testFirst the first instance to be excluded
     * @param testInstances the number of instances to be excluded
     * @return returns the new dataset
     */
    public static Dataset getTrainDataset(Dataset source, int testFirst, int testInstances)
    {
        Instances train = new Instances(source.getInstances(), source.numInstances() - testInstances);
        
        for(int i = 0; i < source.numInstances(); i++)
        {
            if(i < testFirst || i > (testFirst + testInstances - 1))
                train.add(source.getInstances().get(i));
        }
        
        return new Dataset(train);
    }
    
    /**
     * Performs one run of training and testing and returns the evaluation.
     * 
     * @param trainingSet  - the data to train the model on
     * @param testSet      - the data to test the model on
     * @param learningRate - the learning rate for the model
     * @param momentum     - the momentum for the model
     * @param reductMode   - the type of discernibility matrix to use (1 - mAll or 2 - mDec)
     * @return             - returns the evaluation for the model
     */
    public static Evaluation testOnceClassify(Dataset trainingSet, Dataset testSet, float learningRate, float momentum, int reductMode)
    {
        Evaluation eval;
        MultilayerPerceptron neuralNetwork;
        
        // Preprocess data using reduct with largest reduction in dimensionality
        //TODO: Remove timing
        System.out.println("Beginning discretization and reduct selection...");
        long startReduct = System.nanoTime();
        BitSet reductBitSet = findReducts(trainingSet, trainingSet.getName() + "_discretized.arff", reductMode);
        long endReduct = System.nanoTime();
        System.out.println("Time to discretize and find reduct: " + (endReduct - startReduct)/1000000);
        
        // Remove attributes from training set and test set according to reduct
        try
        {
            trainingSet = applyReduct(reductBitSet, trainingSet);
            testSet = applyReduct(reductBitSet, testSet);
        }
        catch(Exception e)
        {
            System.err.println("Could not apply reducts: ");
            e.printStackTrace();
            return null;
        }
        
        neuralNetwork = new MultilayerPerceptron();
        neuralNetwork.setLearningRate(learningRate);
        neuralNetwork.setMomentum(momentum);
        
        // Build and evaluate model based on training data
        try
        {
            neuralNetwork.setHiddenLayers(""+ trainingSet.numAttributes() + "," + trainingSet.numAttributes()/2);
            neuralNetwork.buildClassifier(trainingSet.getInstances());
        }
        catch(Exception e)
        {
            System.err.println("Could not run classifier on training data: " + e.getMessage());
            return null;
        }
        
        //  Test and evaluate model on testing data
        try
        {
            eval  = new Evaluation(testSet.getInstances());
            eval.evaluateModel(neuralNetwork, testSet.getInstances());
            System.out.println("Testing evaluation: " + eval.pctCorrect() + "% Correct");
            System.out.println("                    " + eval.pctIncorrect() + "% Incorrect");
            return eval;
        }
        catch(Exception e)
        {
            System.err.println("Could not run classifier on test set: ");
            e.printStackTrace();
            return null;
        }
    }
    
    /**
     * Prints the proper usage for this program to stdout
     */
    public static void printProperUsage()
    {
        System.out.println("Usage: jml-classifier [mode] [args...]");
        System.out.println("\nwhere modes include:");
        System.out.println("    split <dataset> <folds> <savepath>");
        System.out.println("          split a dataset into separate folds with similar class ratios");
        System.out.println("          dataset  : path to .arff file containing the target dataset");
        System.out.println("          folds    : integer representing the desired number of folds");
        System.out.println("          savepath : base path to save .arff files to\n");
        System.out.println("    test-once <train> <test> <results> <learning-rate> <momentum> <reduct-mode>");
        System.out.println("          builds and trains a neural network on a training set and evaluates");
        System.out.println("          the model on the given testing set. Confusion matrix data is saved");
        System.out.println("          in plain-text to the specified path");
        System.out.println("          train         : path to the .arff file containing the training data");
        System.out.println("          test          : path to the .arff file containing the testing data");
        System.out.println("          results       : path to save the confusion matrix data to");
        System.out.println("          learning-rate : the learning rate for backpropagation (0.0 - 1.0)");
        System.out.println("          momentum      : the momentum coefficient for backpropagation");
        System.out.println("          reduct-mode   : the mode for reduct selection {1, 2}");
        System.out.println("                             1: Use discrenibility matrix of type M-All");
        System.out.println("                             2: Use discernibility matrix of type M-Dec\n");
        System.out.println("    multi <dataset> <savepath> <results> <folds> <learning-rate> <momentum> <reduct-mode>");
        System.out.println("          run cross-validation using concurrent threads");
        System.out.println("          dataset       : path to .arff file containing the target dataset");
        System.out.println("          savepath      : base path to save .arff files to");
        System.out.println("          results       : base path to save confusion matrix data to ");
        System.out.println("          folds         : integer representing the desired number of folds");
        System.out.println("          learning-rate : the learning rate for backpropagation (0.0 - 1.0)");
        System.out.println("          momentum      : the momentum coefficient for backpropagation");
        System.out.println("          reduct-mode   : the mode for reduct selection {1, 2}");
        System.out.println("    help");
        System.out.println("          displays usage information");
        System.out.println("Author: Jordan Moreno-Lacalle");
    }
    
    /**
     * Applies the given reduct to the given dataset and returns the new dataset.
     * All attributes that are not included in the given reduct are removed.
     * 
     * @param reduct BitSet representing the attributes that are included in the reduct
     * @param dataset Dataset containing the data to be modified
     * @return Returns the new dataset having attributes removed
     * @throws Exception
     */
    public static Dataset applyReduct(BitSet reduct, Dataset dataset) throws Exception
    {
        // The indices to be kept
        int indices[] = new int[reduct.cardinality() + 1];

        int index = 0;
        
        // Add reduct attribute indices to indices array
        for(int i = 0; i < dataset.numAttributes(); i++)
        {
            if(reduct.get(i) || i == dataset.classIndex())
            {
                    indices[index] = i;
                    index++; 
            }
        }
        
        /**
         * Remove all attributes that are not part of the reduct
         */
        Remove removeReduct = new Remove();
        removeReduct.setAttributeIndicesArray(indices);
        // Invert selection because we want to KEEP the attributes in indices
        removeReduct.setInvertSelection(true);
        removeReduct.setInputFormat(dataset.getInstances());
        Dataset modifiedData = new Dataset(Filter.useFilter(dataset.getInstances(), removeReduct));
        modifiedData.setClassIndex(modifiedData.numAttributes() - 1);
        
        return modifiedData;
    }
    
    /**
     * Construct a single training dataset from a collection of folds derived from a single set.
     * The training folds set excludes the fold to be tested. 
     * 
     * @param trainFolds
     * @return
     */
    public static Dataset makeTrainingSet(ArrayList<Dataset> trainFolds)
    {
        Instances trainingSet;
        
        // Initialize trainingSet using attributes from a training fold
        trainingSet = new Instances(trainFolds.get(0).getInstances(), 0);
        
        // Add all instances from all folds to trainingSet
        for(Dataset fold : trainFolds)
        {
            for(int i = 0; i < fold.numInstances(); i++)
            {
                trainingSet.add(fold.getInstances().instance(i));
            }
        }
        
        return new Dataset(trainingSet);
    }
    
    /**
     * Splits the given dataset across several folds. The data in each fold is saved
     * to path_fold_i.arff where i is the fold number and path is the given base
     * path.
     * 
     * @param dataset  The dataset to be split
     * @param numFolds The number of folds
     * @param path     The base path to save fold data to
     * @return returns paths to fold files
     */
    public static String[] splitData(DatasetSplitter splitter, Dataset dataset, int numFolds, String path, boolean saveCombined)
    {
        String paths[] = new String[numFolds];
        splitter = new DatasetSplitter(dataset.getInstances());

        splitter.initFolds(numFolds);
        System.out.print("Splitting dataset...");
        splitter.splitData();
        System.out.println("Done.");
        
        if(saveCombined)
            System.out.println("Saving individual and combined files...");
        else
            System.out.println("Saving individual files only...");
        
        ArrayList<Dataset> folds = splitter.getFolds();
        
        /**
         * Save each fold's data individually AND combined datasets
         */
        for (int i = 0; i < folds.size(); i++)
        {
            String foldPath = path + "_fold_" + i + ".arff";
            
            try
            {
                DataSink.write(foldPath, folds.get(i).getInstances());
                paths[i] = foldPath;
                //folds.get(i).loadData(foldPath);
                if(saveCombined)
                {
                    ArrayList<Dataset> foldsCopy = new ArrayList<>(folds);
                    foldsCopy.remove(i);
                    
                    DataSink.write(path + "_combined_excludes_" + i + ".arff", makeTrainingSet(foldsCopy).getInstances());
                }
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
        }
        
        return paths;
    }

    /**
     * Sets the class index of the given dataset and discretizes the data. The
     * resulting dataset is saved to the given path.
     * 
     * @param dataset
     * @param classIndex
     * @param path
     */
    public static void discretize(Dataset dataset, int classIndex, String path)
    {

        try
        {
            dataset.data.setClassIndex(classIndex);
            dataset.discretize(path);
        } catch (Exception e)
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    /**
     * Finds and returns the most minimal reduct for the given dataset
     * 
     * @param dataset The dataset to find a reduct for
     * @param discPath The path to save the discretized data to 
     * @return BitSet representing the most minimal reduct
     */
    public static BitSet findReducts(Dataset dataset, String discPath, int reductMode)
    {
        // Discretize the data and save to a new file, this file will be loaded again and used to compute reducts
        long discStart = System.nanoTime();
        if(null == dataset.discretize(discPath))
        {
            System.err.println("Could not discretize data");
            return null;
        }
        long discEnd = System.nanoTime();
        System.out.println("Discretization took: " + (discEnd - discStart)/1000000);
        /*
         * rseslib uses a different file format than WEKA, so data loading must be
         * handled separately for reduct selection.
         */
        try
        {
            // Setup console output for rseslib methods
            StandardOutput consoleStd = new StandardOutput();
            StandardErrorOutput consoleErr = new StandardErrorOutput();
            Report.addInfoOutput(consoleStd);
            Report.addErrorOutput(consoleErr);

            // Prepare reduct provider with appropriate properties
            Properties properties = new Properties();
            InputStream fileStream;

            //if(reductMode == 1)
            //    fileStream = TransactionClassifier.class.getResourceAsStream("/discernibility-matrix-all.properties");
            //else
            if(reductMode == 2)
            {
                fileStream = TransactionClassifier.class.getResourceAsStream("/discernibility-matrix-dec.properties");
                properties.load(fileStream);
            }
            else
            {
                properties = null;
            }
                            
            DoubleDataTable table = new ArrayListDoubleDataTable(new File(discPath), new StdOutProgress());
            //AllGlobalReductsProvider reductsProvider = new AllGlobalReductsProvider(null, table);
            AllGlobalReductsProvider reductsProvider = new AllGlobalReductsProvider(properties, table);
            
            // Get reducts
            System.out.println("Finding reducts...");
            long reductStart = System.nanoTime();
            Collection<BitSet> reducts = reductsProvider.getReducts();
            long reductEnd = System.nanoTime();
            System.out.println("THREAD: " + Thread.currentThread().getId() + " -Finding reducts took: " + (reductEnd - reductStart)/1000000);
            
            // Get the first reduct (it offers the most reduction in dimensionality)
            Object reductsArray[] = reducts.toArray();
            ArrayList<BitSet> reductsList = new ArrayList<BitSet>();
            reductsList.add((BitSet) reductsArray[0]);
            BitSet firstReduct = reductsList.get(0);

            // Print all selected reduct and all possible reducts
            System.out.println(firstReduct);
            Report.displaynl(reducts);
            Report.close();
            System.out.println();
            
            return firstReduct;
        } 
        catch (Exception e)
        {
            System.out.println("Could not compute reducts");
            System.err.println(e.getMessage());
            return null;
        }
    }
}