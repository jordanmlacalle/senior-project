package com.jordanml.TransactionClassifier;

public class TransactionClassifier {
    
    public static void main(String[] args) {
        Dataset allTransactions = new Dataset("src/creditcard_nom.arff");
        
        if(allTransactions.data == null) {
            System.out.println("No dataset loaded");
        } else {
            System.out.println(allTransactions.getPath() + " has...");
            System.out.println(allTransactions.data.numInstances() + " instances");
            System.out.println(allTransactions.data.numAttributes() + " attributes");
        }
    }
}
