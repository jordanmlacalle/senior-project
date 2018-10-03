package com.jordanml.TransactionClassifier;

import java.util.ArrayList;
import java.util.Random;
import java.lang.Float;

import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Handles the splitting of a dataset across multiple folds.
 * Preserves a similar class ratio across the folds with a similar number
 * of instances of each possible class.
 * 
 */
public class DatasetSplitter
{

    /**
     * folds - collection of Datasets split from source sourceData - the source data
     * sourceData - copy of the data from the target dataset to be split
     * template - the template from which each fold dataset gets its attribute information
     */
    private ArrayList<Dataset> folds;
    private Instances sourceData;
    private Instances template;

    /**
     * needed - the number of instances needed by each fold (set once) 
     * fitnesses - how fit each fold is for selection (how many more instances it needs for each class)
     * probabilities - the probability that a fold should be selected (for each class)
     * foldsInitialized - whether or not the folds have been initialized
     * numClasses - the number of possible classes in the source data
     * classIndex - the attribute index of the class attribute
     * possibleClasses - array of the possible class values
     */
    private int instancesRequired[][];
    private int fitnesses[][];
    private float probabilities[][];
    private boolean foldsInitialized;
    private int numClasses;
    private int classIndex;
    private String possibleClasses[];
    
    public DatasetSplitter(Dataset source)
    {
        foldsInitialized = false;
        sourceData = new Instances(source.data);
        sourceData.setClassIndex(sourceData.numAttributes() - 1);
        template = new Instances(sourceData, 0);
        numClasses = sourceData.classAttribute().numValues();
        classIndex = sourceData.classIndex();
        possibleClasses = new String[numClasses];
        folds = new ArrayList<Dataset>();
        
        
        for(int i = 0; i < numClasses; i++)
        {
            possibleClasses[i] = sourceData.classAttribute().value(i);
        }
    }

    /**
     * Initializes the folds array with the given number of empty Datasets and
     * initializes the
     * 
     * @param numFolds
     */
    public void initFolds(int numFolds)
    {

        // Create new Dataset objects
        for (int i = 0; i < numFolds; i++)
        {
            Dataset currentFold = new Dataset(template);
            folds.add(currentFold);
        }

        instancesRequired = new int[numClasses][numFolds];
        // initialize fitnesses
        fitnesses = new int[numClasses][numFolds];
        // initialize probabilities
        probabilities = new float[numClasses][numFolds];

        foldsInitialized = true;
    }

    /**
     * Splits the source data into the separate folds. Returns true upon completion,
     * false if the process could not begin.
     */
    public boolean splitData()
    {

        //Check if the folds have been initialized, if not then data splitting cannot continue
        if (!isInitialized())
        {
            return false;
        }

        /**
         * numFolds - the number of folds 
         * fitnessSum - the sum of the fitnesses for the current class
         * random - a random number generator used when selecting which fold the current instance will be added to
         * rand - the random number
         * selectedFold - the fold to add the current instance to
         * currentInstance - the instance from the source dataset that is currently under consideration
         */
        int numFolds = folds.size();
        int fitnessSum = 0;
        Random random = new Random();
        float rand;
        int selectedFold = -1;
        int currentInstance = 0;

        // Set the number of instances of each class that each fold expects when splitting is done
        setInstancesRequired();

        // Loop until all Instances have been distributed
        while (currentInstance < sourceData.numInstances())
        {
            // Get copy of current instance
            Instance copy = (Instance) sourceData.instance(currentInstance).copy();
            
            // Get class value index
            int classValueIndex = getClassValueIndex(copy);
            
            // Set the fitnesses
            setFitnesses(classValueIndex);

            // Sum all fitnesses for the current class value
            for(int i = 0; i < numFolds; i++)
            {
                fitnessSum += fitnesses[classValueIndex][i];
            }
              
            // Set probabilities based on fitnessSum for the current class value
            setProbabilities(classValueIndex, fitnessSum);

            // Get new random number
            rand = random.nextFloat();
            
            int i = 0;
            // Find the fold for which the random number falls in the probability range
            while (selectedFold < 0)
            {
                int result = Float.compare(rand, probabilities[classValueIndex][i]);
                if (result < 0 || result == 0)
                {
                    selectedFold = i;
                }
                i++;
            }
            
            // Add instance to selected fold
            folds.get(selectedFold).addInstance(copy);
            // Reset selectedFold, fitnessSum
            selectedFold = -1;
            fitnessSum = 0;
            currentInstance++;
        }
        
        return true;
    }

    /**
     * Sets the fitnesses for each fold. Fitness here is considered to be how many
     * more instances a fold requires. 
     * 
     * @param classValueIndex the index for the class value of the current instance
     */
    private void setFitnesses(int classValueIndex)
    {        
        Instances foldData;
        int classCounts[];
        
        for(int i = 0; i < folds.size(); i++)
        {
            // Get data from current fold
            foldData = folds.get(i).data;
            // Get counts of each class
            classCounts = foldData.attributeStats(classIndex).nominalCounts;
            // Fitness = required instances of current class - instances had from current class
            fitnesses[classValueIndex][i] = instancesRequired[classValueIndex][i] - classCounts[classValueIndex];
        }
    }

    /**
     * Sets the probabilities for each fold. The probability determines the
     * likeliness that a fold is selected.
     * 
     * @param sum The sum of the fitnesses for all folds
     */
    private void setProbabilities(int classValueIndex, int sum)
    {
        int partialSum = 0;
        int numFolds = folds.size();
        
        for (int i = 0; i < numFolds; i++)
        {
            partialSum += fitnesses[classValueIndex][i];
            probabilities[classValueIndex][i] = (float) partialSum / (float) sum;
        }
    }

    /**
     * Sets the number of instances required by each fold
     */
    private void setInstancesRequired()
    {
        
        // Get count of each class
        AttributeStats stats = sourceData.attributeStats(classIndex);
        int classCounts[] = stats.nominalCounts;
        
        // Get number of folds
        int numFolds = folds.size();
        
        // For each class...
        for(int i = 0; i < numClasses; i++)
        {
            // Initial calculation of minimum number of instances (of the current class) per fold
            int minPerFold = classCounts[i] / numFolds;
            // Instances of this class remaining after initial calculation
            int remaining = classCounts[i] % numFolds;
            
            for(int j = 0; j < numFolds; j++)
            {
                instancesRequired[i][j] = minPerFold; 
                
                // If unassigned instances of this class remain, add one
                if(remaining > 0)
                {
                    instancesRequired[i][j]++;
                    remaining--;
                }
            }
        }
    }

    /**
     * Gets the class value index by comparing the String value of the current
     * instance's class to the possible class values.
     * 
     * @param currentInstance The current instance being considered
     * @return integer value representing the class value index
     */
    private int getClassValueIndex(Instance currentInstance)
    {
        String classValue = currentInstance.stringValue(classIndex);
        int classValueIndex = -1;
        
        for(int i = 0; i < numClasses && classValueIndex < 0; i++)
        {
            if(classValue.equals(possibleClasses[i]))
                classValueIndex = i;
        }
        
        return classValueIndex;
    }
    /**
     * Returns the folds ArrayList
     * 
     * @return the ArrayList that contains each dataset created by splitting the
     *         source dataset
     */
    public ArrayList<Dataset> getFolds()
    {
        return folds;
    }

    /**
     * Returns the Instances object containing the source data
     * 
     * @return the Instances object containing the source data
     */
    public Instances getSource()
    {
        return sourceData;
    }

    /**
     * Getter to check whether or not folds have been initialized and data is ready
     * to be split
     * 
     * @return boolean. True if folds have been initialized and data is ready to be
     *         split.
     */
    public boolean isInitialized()
    {
        return foldsInitialized;
    }

}
