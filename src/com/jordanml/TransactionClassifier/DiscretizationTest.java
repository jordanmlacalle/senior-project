package com.jordanml.TransactionClassifier;

import weka.core.converters.ConverterUtils.DataSink;

public class DiscretizationTest {

    public static void main(String[] args) {
        Dataset allTransactions = new Dataset("src/data/creditcard_nom.arff");
        
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
    }

}
