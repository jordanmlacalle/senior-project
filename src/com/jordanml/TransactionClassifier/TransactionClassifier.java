package com.jordanml.TransactionClassifier;

import java.io.File;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collection;

// RSESLIB
import rseslib.processing.reducts.AllGlobalReductsProvider;
import rseslib.structure.table.ArrayListDoubleDataTable;
import rseslib.structure.table.DoubleDataTable;
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

    public static final int POSITIVE_CLASS_INDEX = 1;
    
    public static void main(String[] args)
    {
        long start = System.nanoTime(); // Time that execution started
        String filepath = "../data/creditcard_nom.arff";
        Dataset testData = new Dataset(filepath);
        testData.setClassIndex(testData.numAttributes() - 1);
        final int numFolds = 10;

        // Ensure that dataset was loaded
        if (!testData.hasData())
        {
            System.out.println("No dataset loaded");
            return;
        } 
        else
        {
            System.out.println(testData.getPath() + " has...");
            System.out.println(testData.numInstances() + " instances");
            System.out.println(testData.numAttributes() + " attributes");
        }

        classifierTrial(testData, testData.numAttributes()-1, numFolds);
        long end = System.nanoTime(); // Time that execution ended
        System.out.println("Terminating: " + (end - start)/1000000);
    }

    /**
     * 
     * @param dataset the source data
     * @param classIndex the index to the class attribute
     * @param numFolds the number of folds to use in cross validation
     * 
     * @throws Exception
     */
    public static void classifierTrial(Dataset dataset, int classIndex, int numFolds)
    {
        dataset.setClassIndex(classIndex);
        
        // TEMPORARY - time this
        long start = System.nanoTime();
        String foldPaths[] = splitData(dataset, numFolds, "../data/test/testData");
        long end = System.nanoTime();
        System.out.println("Splitting took: " + (end-start)/1000000);
        double totalTP = 0;
        double totalFP = 0;
        double totalTN = 0;
        double totalFN = 0;
        
        // 10-CV loop tasks:
        /**
         * For each fold:
         *      - Designate current fold as test fold
         *      - Discretize all folds (including or excluding test fold?)
         *      - Find reducts for folds
         *      - Train on 9 training folds
         *      - Test on testing fold
         *      
         */
        for(int i = 0; i < numFolds; i++)
        {
            System.out.println("\nBuilding model to test on fold " + i + ":");
            
            Dataset testSet = new Dataset(foldPaths[i]);
            testSet.setClassIndex(classIndex);
            Dataset trainingSet;
            ArrayList<Instances> trainFolds = new ArrayList<Instances>();
            MultilayerPerceptron neuralNetwork = new MultilayerPerceptron();
            neuralNetwork.setLearningRate(0.2);
            neuralNetwork.setTrainingTime(1000);
            
            /**
             * eval will run classifier on a dataset and generate statistics 
             * (false positives, true positives, etc. that can be used for confusion matrix)
             */
            Evaluation eval;
            
            // Load training folds and add them to the ArrayList
            for(int j = 0; j < numFolds; j++)
            {
                if(j != i)
                {
                    Dataset newTrainFold = new Dataset(foldPaths[j]);
                    newTrainFold.setClassIndex(classIndex);
                    trainFolds.add(newTrainFold.data);
                }
            } 
            
            // Merge training folds into single training set
            // Instances trainingSet = makeTrainingSet(trainFolds);
            trainingSet = new Dataset(makeTrainingSet(trainFolds));
            trainFolds.clear();
            trainingSet.saveFile("../data/trainingSet_" + i + ".arff");
            // Preprocess data using reduct with largest reduction in dimensionality
            //TODO: Remove timing
            long startReduct = System.nanoTime();
            BitSet reductBitSet = findReducts(trainingSet, "../data/disc_trainingSet_" + i + ".arff");
            long endReduct = System.nanoTime();
            System.out.println("Time to discretize and find reduct: " + (endReduct - startReduct)/1000000);
            trainingSet.saveFile("../data/trainingSet_" + i + "_afterDisc.arff");
            
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
                return;
            }
            
            // Build and evaluate model based on training data
            try
            {
                eval = new Evaluation(trainingSet.getInstances());
                neuralNetwork.setHiddenLayers(""+ trainingSet.numAttributes() + "," + trainingSet.numAttributes()/2);
                neuralNetwork.buildClassifier(trainingSet.getInstances());
            }
            catch(Exception e)
            {
                System.err.println("Could not run classifier on training data: " + e.getMessage());
                return;
            }
            
            //  Test and evaluate model on testing data
            try
            {
                eval  = new Evaluation(testSet.getInstances());
                eval.evaluateModel(neuralNetwork, testSet.getInstances());
                System.out.println("Testing evaluation - Fold " + i + ": " + eval.pctCorrect() + "% Correct");
                System.out.println("                             " + eval.pctIncorrect() + "% Incorrect");
                totalTP += eval.numTruePositives(POSITIVE_CLASS_INDEX);
                totalFP += eval.numFalsePositives(POSITIVE_CLASS_INDEX);
                totalTN += eval.numTrueNegatives(POSITIVE_CLASS_INDEX);
                totalFN += eval.numFalseNegatives(POSITIVE_CLASS_INDEX);

            }
            catch(Exception e)
            {
                System.err.println("Could not run classifier on test set: ");
                e.printStackTrace();
                return;
            }
        }   
        
        // Build confusion matrix
        System.out.println(" True Positives: " + totalTP);
        System.out.println("False Positives: " + totalFP);
        System.out.println(" True Negatives: " + totalTN);
        System.out.println("False Negatives: " + totalFN);
        System.out.println("       Accuracy: " + (totalTP + totalTN) / (totalTP + totalTN + totalFP + totalFN));
        System.out.println("   Predictivity: " + (totalTP / (totalTP + totalFP)));
        System.out.println("    Selectivity: " + (totalTN / (totalTN + totalFP)));
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
    public static Instances makeTrainingSet(ArrayList<Instances> trainFolds)
    {
        Instances trainingSet;
        
        // Initialize trainingSet using attributes from a training fold
        trainingSet = new Instances(trainFolds.get(0), 0);
        
        // Add all instances from all folds to trainingSet
        for(Instances fold : trainFolds)
        {
            for(int i = 0; i < fold.numInstances(); i++)
            {
                trainingSet.add(fold.instance(i));
            }
        }
        
        return trainingSet;
    }
    
    /**
     * Splits the given dataset across several folds. The data in each fold is saved
     * to path_fold_i.arff where i is the fold number and path is the given base
     * path.
     * 
     * @param dataset  The dataset to be split
     * @param numFolds The number of folds
     * @param path     The base path to save fold data to
     */
    public static String[] splitData(Dataset dataset, int numFolds, String path)
    {
        String paths[] = new String[numFolds];
        DatasetSplitter splitter = new DatasetSplitter(dataset.getInstances());

        splitter.initFolds(numFolds);
        splitter.splitData();
        ArrayList<Dataset> folds = splitter.getFolds();

        /**
         * Save each fold's data
         */
        for (int i = 0; i < folds.size(); i++)
        {
            String foldPath = path + "_fold_" + i + ".arff";
            try
            {
                DataSink.write(foldPath, folds.get(i).getInstances());
                paths[i] = foldPath;
                //folds.get(i).loadData(foldPath);
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
    public static BitSet findReducts(Dataset dataset, String discPath)
    {
        // Discretize the data and save to a new file, this file will be loaded again and used to compute reducts
        if(null == dataset.discretize(discPath))
        {
            System.err.println("Could not discretize data");
            return null;
        }
        
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

            // Prepare reduct provider
            DoubleDataTable table = new ArrayListDoubleDataTable(new File(discPath), new StdOutProgress());
            AllGlobalReductsProvider reductsProvider = new AllGlobalReductsProvider(null, table);

            // Get reducts
            Collection<BitSet> reducts = reductsProvider.getReducts();

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
    
    
    /**
     * Build the confusion matrix from evaluation data for all loops of cross validation
     * 
     * @param modelEvaluations
     * @param positiveClassIndex
     */
    private static void computeConfusionMatrix(ArrayList<Evaluation> modelEvaluations, int positiveClassIndex)
    {
        double totalTP = 0;
        double totalFP = 0;
        double totalTN = 0;
        double totalFN = 0;
        
        for(Evaluation eval : modelEvaluations)
        {
            totalTP += eval.numTruePositives(positiveClassIndex);
            totalFP += eval.numFalsePositives(positiveClassIndex);
            totalTN += eval.numTrueNegatives(positiveClassIndex);
            totalFN += eval.numFalseNegatives(positiveClassIndex);
        }
        
        System.out.println(" True Positives: " + totalTP);
        System.out.println("False Positives: " + totalFP);
        System.out.println(" True Negatives: " + totalTN);
        System.out.println("False Negatives: " + totalFN);
        //System.out.println("       Accuracy: " + (totalTP + totalTN) / (totalFP + totalFN));
        System.out.println("   Predictivity: " + (totalTP / (totalTP + totalFP)));
        System.out.println("    Selectivity: " + (totalTN / (totalTN + totalFP)));
    }
}