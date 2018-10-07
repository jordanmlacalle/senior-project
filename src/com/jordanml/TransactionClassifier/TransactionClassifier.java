package com.jordanml.TransactionClassifier;

import java.io.File;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collection;

import rseslib.processing.reducts.AllGlobalReductsProvider;

import rseslib.structure.table.ArrayListDoubleDataTable;
import rseslib.structure.table.DoubleDataTable;
import rseslib.system.Report;
import rseslib.system.output.StandardErrorOutput;
import rseslib.system.output.StandardOutput;
import rseslib.system.progress.StdOutProgress;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;

public class TransactionClassifier
{

    public static void main(String[] args)
    {

        String filepath = "../data/credit-g.arff";
        Dataset testData = new Dataset(filepath);
        final int numFolds = 10;

        // Ensure that dataset was loaded
        if (!testData.hasData())
        {
            System.out.println("No dataset loaded");
            return;
        } else
        {
            System.out.println(testData.getPath() + " has...");
            System.out.println(testData.numInstances() + " instances");
            System.out.println(testData.numAttributes() + " attributes");
        }

        try
        {
            classifierTrial(testData, testData.numAttributes()-1, numFolds);
        } catch (Exception e)
        {
            System.err.println("Classifier failed");
            e.printStackTrace();
        }

        System.out.println("Terminating");
    }

    /**
     * 
     * @param dataset the source data
     * @param classIndex the index to the class attribute
     * @param numFolds the number of folds to use in cross validation
     * 
     * @throws Exception
     */
    public static void classifierTrial(Dataset dataset, int classIndex, int numFolds) throws Exception
    {
        dataset.setClassIndex(classIndex);
        String foldPaths[] = splitData(dataset, numFolds, "../data/test/testData");
        ArrayList<Evaluation> modelEvaluations = new ArrayList<Evaluation>();
        
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
            
            Dataset testFold = new Dataset(foldPaths[i]);
            testFold.setClassIndex(classIndex);
            ArrayList<Instances> trainFolds = new ArrayList<Instances>();
            MultilayerPerceptron neuralNetwork = new MultilayerPerceptron();
            
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
            Instances trainingSet = makeTrainingSet(trainFolds);
            // Build and evaluate model based on training data
            neuralNetwork.buildClassifier(trainingSet);
            eval  = new Evaluation(trainingSet);
            eval.evaluateModel(neuralNetwork, trainingSet);
            System.out.println("Training evaluation - Fold " + i + ": " + eval.pctCorrect() + "% Correct");
            System.out.println("                              " + eval.pctIncorrect() + "% Incorrect");
            
            //  Test and evaluate model on testing data
            eval  = new Evaluation(testFold.data);
            eval.evaluateModel(neuralNetwork, testFold.data);
            modelEvaluations.add(eval);
            System.out.println("Testing evaluation - Fold " + i + ": " + eval.pctCorrect() + "% Correct");
            System.out.println("                             " + eval.pctIncorrect() + "% Incorrect");
        }   
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
        DatasetSplitter splitter = new DatasetSplitter(dataset);

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
                DataSink.write(foldPath, folds.get(i).data);
                paths[i] = foldPath;
                //folds.get(i).loadData(foldPath);
            } catch (Exception e)
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

    public static void findReducts(Dataset dataset)
    {

        if (dataset.getPath().equals(null))
        {
            return;
        }
        /*
         * rseslib uses a different file format than WEKA, so data loading must be
         * handled separately for reduct selection.
         */
        try
        {
            /*
             * Setup console output for rseslib methods
             */
            StandardOutput consoleStd = new StandardOutput();
            StandardErrorOutput consoleErr = new StandardErrorOutput();
            Report.addInfoOutput(consoleStd);
            Report.addErrorOutput(consoleErr);

            /*
             * Prepare reduct provider
             */
            DoubleDataTable table = new ArrayListDoubleDataTable(new File(dataset.getPath()), new StdOutProgress());
            AllGlobalReductsProvider reductsProvider = new AllGlobalReductsProvider(null, table);

            /*
             * Get reducts
             */
            Collection<BitSet> reducts = reductsProvider.getReducts();

            // Test getting a single reduct
            Object reductsArray[] = reducts.toArray();
            ArrayList<BitSet> intArray = new ArrayList<BitSet>();
            intArray.add((BitSet) reductsArray[0]);
            BitSet currentReduct = intArray.get(0);

            // Print all reducts
            System.out.println(currentReduct);
            Report.displaynl(reducts);
            Report.close();
            System.out.println();

        } catch (Exception e)
        {
            System.out.println("Could not compute reducts");
            System.err.println(e.getMessage());
        } finally
        {
            System.out.println("Computed reducts");
        }
    }
}