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
    private int needed[];
    private int fitnesses[];
    private float probabilities[];
    
    public DatasetSplitter(Dataset source) {
        folds = new ArrayList<Dataset>();
        sourceData = new Instances(source.data);
        template = new Instances(sourceData, 0);
    }
    
    /**
     * 
     * @param numFolds
     */
    public void initFolds(int numFolds) {
        
        // Create new Dataset objects
        for(int i = 0; i < numFolds; i++) {
            
            Dataset currentFold = new Dataset(template);
            folds.add(currentFold);
        }
        
        needed = new int[numFolds];
        //initialize fitnesses
        fitnesses = new int[numFolds];
        //initialize probabilities
        probabilities = new float[numFolds];
        
    }
    
    /**
     * 
     */
    public void splitData() {
        
        int numFolds = folds.size();
        int fitnessSum = 0;
        Random random = new Random();
        float rand;
        int selectedFold  = -1;
        int currentInstance = 0;
        
        setNeeded();
        
        while(currentInstance < sourceData.numInstances()) {
            
            System.out.println("Current Instance: " + currentInstance);
            System.out.println("Num Instances: " + sourceData.numInstances());
            setFitnesses();
            
            for(int i = 0; i < numFolds; i++) {
                fitnessSum += fitnesses[i];
            }

            setProbabilities(fitnessSum);
            
            rand = random.nextFloat();
            int i = 0; 
            
            while(selectedFold < 0) {
                int result = Float.compare(rand, probabilities[i]);
                if(result < 0 || result == 0) {
                    selectedFold = i;
                }
                i++;
            }
            
            Instance copy = (Instance) sourceData.instance(currentInstance).copy();
            folds.get(selectedFold).addInstance(copy);
            selectedFold = -1;
            fitnessSum = 0;
            currentInstance++;
            
        }
    }
    
    /**
     * Sets the fitnesses for each fold. Fitness here is considered to be 
     * how many more instances a fold requires.
     */
    public void setFitnesses() {
        
        for(int i = 0; i < folds.size(); i++) {
            fitnesses[i] = needed[i] - folds.get(i).numInstances();
        }
    }
    
    /**
     * Sets the probabilities for each fold. The probability determines the likeliness 
     * that a fold is selected.
     * @param sum  The sum of the fitnesses for all folds
     */
    public void setProbabilities(int sum) {
        
        int partialSum = 0;
        
        for(int i = 0; i < folds.size(); i++) {
            partialSum += fitnesses[i];
            probabilities[i] = (float) partialSum / (float) sum;
        }
    }
    
    /**
     * Sets the number of instances that each fold will have.
     */
    public void setNeeded() {
        
        int numFolds = folds.size();
        int minPerFold = sourceData.numInstances() / numFolds;
                
        for(int i = 0; i < folds.size()-1; i++) {
            needed[i] = minPerFold;
        }
        
        needed[folds.size()-1] = sourceData.numInstances() - (numFolds-1) * minPerFold;
        
    }
    
    /**
     * Returns the folds ArrayList 
     * @return folds, the ArrayList<Dataset> that contains each dataset 
     *         created by splitting the source dataset
     */
    public ArrayList<Dataset> getFolds() {
        return folds;
    }


}
