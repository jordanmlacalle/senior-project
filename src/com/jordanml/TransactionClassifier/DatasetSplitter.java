package com.jordanml.TransactionClassifier;

import java.util.ArrayList;
import java.util.Random;
import java.lang.Float;

import weka.core.Instance;
import weka.core.Instances;

public class DatasetSplitter {
    
    /**
     * folds - collection of Datasets split from source
     * sourceData - the source data
     * template - the template from which each fold dataset gets its attribute information
     */
    private ArrayList<Dataset> folds;
    private Instances sourceData;
    private Instances template;
   
    /**
     * needed - the number of instances needed by each fold (set once)
     * fitnesses - how fit each fold is for selection (how many more instances it needs)
     * probabilities - the probability that a fold should be selected
     */
    private int instancesRequired[];
    private int fitnesses[];
    private float probabilities[];
    private boolean foldsInitialized;
    
    public DatasetSplitter(Dataset source) {
        folds = new ArrayList<Dataset>();
        sourceData = new Instances(source.data);
        template = new Instances(sourceData, 0);
        foldsInitialized = false;
    }
    
    /**
     * Initializes the folds array with the given number of empty Datasets
     * and initializes the 
     * @param numFolds
     */
    public void initFolds(int numFolds) {
        
        // Create new Dataset objects
        for(int i = 0; i < numFolds; i++) {
            
            Dataset currentFold = new Dataset(template);
            folds.add(currentFold);
        }
        
        instancesRequired = new int[numFolds];
        //initialize fitnesses
        fitnesses = new int[numFolds];
        //initialize probabilities
        probabilities = new float[numFolds];
        
        foldsInitialized = true;
    }
    
    /**
     *  Splits the source data into the separate folds.
     *  Returns true upon completion, false if the process could not begin.
     */
    public boolean splitData() {
        
        if(!isInitialized()) {
            return false;
        }
        
        int numFolds = folds.size();
        int fitnessSum = 0;
        Random random = new Random();
        float rand;
        int selectedFold  = -1;
        int currentInstance = 0;
        
        setInstancesRequired();
        
        //Loop until all Instances have been added
        while(currentInstance < sourceData.numInstances()) {

            setFitnesses();
            
            //Sum all fitnesses
            for(int i = 0; i < numFolds; i++) {
                fitnessSum += fitnesses[i];
            }
            //Set probabilities based on fitnessSum
            setProbabilities(fitnessSum);
            
            rand = random.nextFloat();
            int i = 0; 
            
            // Find the fold for which the random number falls in the probability range 
            while(selectedFold < 0) {
                int result = Float.compare(rand, probabilities[i]);
                if(result < 0 || result == 0) {
                    selectedFold = i;
                }
                i++;
            }
            //Add Instance to selected fold
            Instance copy = (Instance) sourceData.instance(currentInstance).copy();
            folds.get(selectedFold).addInstance(copy);
            selectedFold = -1;
            fitnessSum = 0;
            currentInstance++;
        }
        
        return true;
    }
    
    /**
     * Sets the fitnesses for each fold. Fitness here is considered to be 
     * how many more instances a fold requires.
     */
    private void setFitnesses() {
        
        for(int i = 0; i < folds.size(); i++) {
            fitnesses[i] = instancesRequired[i] - folds.get(i).numInstances();
        }
    }
    
    /**
     * Sets the probabilities for each fold. The probability determines the likeliness 
     * that a fold is selected.
     * @param sum  The sum of the fitnesses for all folds
     */
    private void setProbabilities(int sum) {
        
        int partialSum = 0;
        
        for(int i = 0; i < folds.size(); i++) {
            partialSum += fitnesses[i];
            probabilities[i] = (float) partialSum / (float) sum;
        }
    }
    
    /**
     * Sets the number of instances required by each fold
     */
    private void setInstancesRequired() {
        
        int numFolds = folds.size();
        int minPerFold = sourceData.numInstances() / numFolds;
                
        //Set the number of Instances per fold, this will be used to compute fitness
        for(int i = 0; i < folds.size()-1; i++) {
            instancesRequired[i] = minPerFold;
        }
        //Set the max number of Instances for the final fold (just push the remaining Instances into this fold)
        instancesRequired[folds.size()-1] = sourceData.numInstances() - (numFolds-1) * minPerFold;
        
    }
    
    /**
     * Returns the folds ArrayList 
     * @return the ArrayList that contains each dataset 
     *         created by splitting the source dataset
     */
    public ArrayList<Dataset> getFolds() {
        return folds;
    }
    
    /**
     * Returns the Instances object containing the source data
     * @return the Instances object containing the source data
     */
    public Instances getSource() {
        return sourceData;
    }
    
    /**
     * Getter to check whether or not folds have been initialized and data is ready to be split
     * @return boolean. True if folds have been initialized and data is ready to be split.
     */
    public boolean isInitialized() {
        return foldsInitialized;
    }


}
