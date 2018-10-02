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
import weka.core.converters.ConverterUtils.DataSink;

public class TransactionClassifier
{

    public static void main(String[] args)
    {

        String filepath = "../data/breast-cancer.arff";
        Dataset testData = new Dataset(filepath);

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

        // Trial run data splitting
        ArrayList<Dataset> folds = splitData(testData, 10, "../data/test/testData");

        // Trial run discretization and reducts, 10-CV can be done in a foreach loop
        // like this
        int i = 0;
        for (Dataset fold : folds)
        {
            discretize(fold, fold.numAttributes() - 1, "../data/test/disc_testFold_" + i + ".arff");
            findReducts(fold);
            i++;
        }

        System.out.println("Terminating");
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
    public static ArrayList<Dataset> splitData(Dataset dataset, int numFolds, String path)
    {

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
                folds.get(i).loadData(foldPath);
            } catch (Exception e)
            {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }

        return folds;
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