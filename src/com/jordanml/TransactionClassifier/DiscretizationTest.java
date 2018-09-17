package com.jordanml.TransactionClassifier;

import weka.core.converters.ConverterUtils.DataSink;

public class DiscretizationTest {

    public static void main(String[] args) {
        Dataset allTransactions = new Dataset("src/data/creditcard_nom.arff");
        
<<<<<<< refs/remotes/origin/#3_Verify_discretization_with_histograms
        if(!checkDataset(allTransactions)) {
            return;
        }
        //discretization requires class index to be set
        allTransactions.data.setClassIndex(allTransactions.numAttributes() - 1);
        
        tryDiscretization(allTransactions);
        
        
        System.out.println("Done.");
    }
    
    public static Boolean checkDataset(Dataset dataset) {
        if(dataset.data == null) {
            System.out.println("No dataset loaded");
            return false;
        } else {
            System.out.println(dataset.getPath() + " has...");
            System.out.println(dataset.numInstances() + " instances");
            System.out.println(dataset.numAttributes() + " attributes");
            return true;
        }
    }
    
    public static void tryDiscretization(Dataset dataset) {
        try {
            DataSink.write("src/data/discretization_test/before.arff", dataset.data);
        } catch (Exception e) {
            System.err.println("Could not write file");
        }
        dataset.discretize("src/data/discretization_test/after.arff");
=======
        if(allTransactions.data == null) {
            System.out.println("No dataset loaded");
        } else {
            System.out.println(allTransactions.getPath() + " has...");
            System.out.println(allTransactions.data.numInstances() + " instances");
            System.out.println(allTransactions.data.numAttributes() + " attributes");
        }
        
        allTransactions.data.setClassIndex(allTransactions.numAttributes() - 1);
        
        try {
            DataSink.write("src/data/discretization_test/before.csv", allTransactions.data);
        } catch (Exception e) {
            System.err.println("Could not write file");
        }
        allTransactions.discretize("src/data/discretization_test/after.csv");
        
        System.out.println("Done.");
>>>>>>> Setup discretization test
    }

}
