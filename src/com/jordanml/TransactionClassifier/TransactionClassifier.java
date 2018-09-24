package com.jordanml.TransactionClassifier;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;

import rseslib.processing.reducts.AllGlobalReductsProvider;
import rseslib.structure.attribute.formats.HeaderFormatException;
import rseslib.structure.data.formats.DataFormatException;
import rseslib.structure.table.ArrayListDoubleDataTable;
import rseslib.structure.table.DoubleDataTable;
import rseslib.system.PropertyConfigurationException;
import rseslib.system.Report;
import rseslib.system.output.StandardErrorOutput;
import rseslib.system.output.StandardOutput;
import rseslib.system.progress.StdOutProgress;

public class TransactionClassifier {
    
    public static void main(String[] args) {
        
        String filepath = "../data/breast-cancer.arff";
        Dataset testData = new Dataset(filepath);
        
        if(!testData.hasData()) {
            System.out.println("No dataset loaded");
            return;
        } else {
            System.out.println(testData.getPath() + " has...");
            System.out.println(testData.numInstances() + " instances");
            System.out.println(testData.numAttributes() + " attributes");
        }
        
        try {
            testData.data.setClassIndex(testData.numAttributes()-1);
            testData.discretize("../data/disc_breast-cancer.arff");
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        /*
         * rseslib uses a different file format than WEKA, so data loading must be
         * handled separately for reduct selection.
         */
        try {
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
            DoubleDataTable table = new ArrayListDoubleDataTable(new File("../data/disc_breast-cancer.arff"), new StdOutProgress());
            AllGlobalReductsProvider reductsProvider = new AllGlobalReductsProvider(null, table);
            
            /*
             * Get reducts
             */
            Collection<BitSet> reducts = reductsProvider.getReducts();
            Report.display(reducts);
            System.out.println();
            
            int i = testData.numAttributes();
            
            for(BitSet bit : reducts) {
                for(int j = 0; j < i; j++) {
                    System.out.println("Bit " + j + ": Value: " + bit.get(j));
                }
            }
            
        } catch (InterruptedException e) {
            System.err.println("Could not load data");
            e.printStackTrace();
        } catch (PropertyConfigurationException e) {
            System.err.println("Could not create reduct provider");
            e.printStackTrace();
        } catch (IOException e) {
            System.err.println("Failed to get reducts");
            e.printStackTrace();
        } catch (HeaderFormatException e) {
            System.err.println("Failed to get reducts");
            e.printStackTrace();
        } catch (DataFormatException e) {
            System.err.println("Failed to get reducts");
            e.printStackTrace();
        } finally {
            System.out.println("Computed reducts");
        }
        
        System.out.println("Terminating");
    }
}
