package com.jordanml.TransactionClassifier;


import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class Dataset {
    
    /**
     * data - the object that contains the dataset 
     * path - the path to the file from which the dataset was loaded
     */
    public Instances data = null;
    private String path = null;
    
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
     * Path getter
     * @return Returns the path to the file containing the dataset
     */
    public String getPath() {
        return path;
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
    
    /**
     * Discretize data using WEKA implementation of Fayyad & Irani MDL discretization.
     * See WEKA 3-8-2 Manual p.219
     */
    public void discretize(Boolean replaceData) {
        
        Instances discretizedData = null;
        Discretize discretizer = new Discretize();
        
        try {
            discretizer.setInputFormat(data);
            discretizedData = Filter.useFilter(data, discretizer);
            
            if(replaceData.equals(true)) {
                data = discretizedData;
            } else {
                //Save to new file
            }
        } catch (Exception e) {
            System.out.println("Error discretizing data");
            System.out.println(e.getMessage());
        }
    }
    
}
