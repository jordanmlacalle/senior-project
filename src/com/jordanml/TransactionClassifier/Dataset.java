package com.jordanml.TransactionClassifier;


import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class Dataset {
    
    /**
     * data - the object that contains the dataset 
     * instances - the number of instances in the dataset
     * attributes - the number of attributes in the dataset
     * path - the path to the file from which the dataset was loaded
     */
    public Instances data;
    private String path;
    
    /**
     * Constructor
     */
    public Dataset() {
        data = null;
        path = null;
    }
    
    /**
     * Constructor with path as parameter. Loads dataset when new Dataset object is created.
     * @param path The path to the file containing the dataset
     * @param classIndex The index of the class attribute ( 0-indexed)
     */
    public Dataset(String path) {
        loadData(path);
    }
    
    /**
     * Getters
     */
    public String getPath() {
        return path;
    }
    
    public int numInstances() {
        return data.numInstances();
    }
    
    public int numAttributes() {
        return data.numAttributes();
    }

    public boolean hasData() {
        if(data == null) {
            return false;
        }
        
        return true;
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
            this.path = path;
            
        } catch (Exception e) {
            System.err.println("Error reading " + path);
            System.err.println(e.getMessage());
        }
    }
    
    /**
     * Discretize data using WEKA implementation of Fayyad & Irani MDL discretization.
     * See WEKA 3-8-2 Manual p.219. 
     * @param savePath Path to save discretized data to. The file extension
     * (.arff or .csv) is specified when providing savePath.
     */
    public boolean discretize(String savePath) throws Exception {
        
        if(data.classIndex() == -1) {
            System.out.println("Class index not set. Set class index prior to discretization.");
            return false;
        }
        
        Instances discretizedData = null;
        Discretize discretizer = new Discretize();
        
        try {
            //discretize data
            discretizer.setInputFormat(data);
            discretizedData = Filter.useFilter(data, discretizer);
            
        } catch (Exception e) {
            System.out.println("Error discretizing data");
            System.err.println(e.getMessage());
            throw e;
        } finally {
            if(savePath == null) {
                //replace data and set path to null 
                data = discretizedData;
                path = null;
            } else {
                /*Save discretized data to new file. See WEKA 3-8-2 Manual p.238.
                 * The DataSink.write method can save to both .arff and .csv
                 */
                try {
                    DataSink.write(savePath, discretizedData);
                } catch (Exception e) {
                    System.err.println("Error saving discretized data to " + savePath);
                    System.err.println(e.getMessage());
                    throw e;
                }    
            }
        } 
        
        return true;
    }
}
