package com.jordanml.TransactionClassifier;

import weka.classifiers.Evaluation;

public class CrossValidationThread extends Thread
{
    private Dataset testSet = null;
    private Dataset trainSet = null;
    private int fold = 0;
    private int reductMode = 1;
    private float learningRate;
    private float momentum;
    private String resultsPath = null;
    
    public void run()
    {
        long t_start = System.nanoTime();
        System.out.println("Thread " + Thread.currentThread().getId() + "is evaluating on fold " + fold);
        Evaluation results = TransactionClassifier.testOnceClassify(trainSet, testSet, learningRate, momentum, reductMode);
        TransactionClassifier.saveResults(results, resultsPath);
        long t_end = System.nanoTime();
        System.out.println("Thread " + Thread.currentThread().getId() + " terminating: " + (t_end - t_start)/1000000 + " ms");
    }
    
    public void init(String path, Dataset train, Dataset test, int foldNum, float learningRate, float momentum, int reductMode)
    {
        testSet = test;
        trainSet = train;
        fold = foldNum;
        resultsPath = path;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.reductMode = reductMode;
    }
}
