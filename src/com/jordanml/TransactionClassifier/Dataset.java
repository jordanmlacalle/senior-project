package com.jordanml.TransactionClassifier;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Dataset {
    
    /**
     * data - the object that contains the dataset 
     * path - the path to the file from which the dataset was loaded
     */
    public Instances data = null;
    public String path = null;
    
    /**
     * Constructor
     */
    public Dataset() {}
    
    /**
     * Constructor with path as parameter. Loads dataset when new Dataset object is created.
     * @param path The path to the file containing the dataset.
     */
    public Dataset(String path) {
        this.path = path;
        loadData(path);
    }
    
    /**
     * Loads dataset from the provided file into the data Instances object.
     * 
     * @param path The path to the file containing the dataset.
     */
    public void loadData(String path) {
        
        System.out.println("Loading data from " + path + "...");
        
        try {
            //See p.210 of the WEKA 3-8-2 manual
            data = DataSource.read(path);
        } catch (Exception e) {
            System.out.println("Error reading " + path);
            System.out.println(e.getMessage());
        }
    }
    
}
