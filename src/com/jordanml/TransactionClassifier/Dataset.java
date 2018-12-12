package com.jordanml.TransactionClassifier;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class Dataset
{

    /**
     * data - the object that contains the dataset instances
     * path - the path to the file from which the dataset was loaded
     * name - the name of the dataset
     */
    public Instances data;
    private String path;
    private String name;
    
    /**
     * Constructor
     */
    public Dataset()
    {
        data = null;
        path = null;
        name = null;
    }

    /**
     * Constructor with path as parameter. Loads dataset when new Dataset object is
     * created.
     * 
     * @param path       The path to the file containing the dataset
     * @param classIndex The index of the class attribute ( 0-indexed)
     * @throws Exception
     */
    public Dataset(String path)
    {
        loadData(path);
    }

    /**
     * Constructor with an Instances object as a parameter. Loads data from an
     * existing instances object.
     * 
     * @param instances The existing Instances object to copy.
     */
    public Dataset(Instances instances)
    {
        loadData(instances);
    }

    /**
     * Getters
     */
    /**
     * Gets path to dataset arff or csv
     * 
     * @return String - path to the dataset arff or csv
     */
    public String getPath()
    {
        return path;
    }

    public String getName()
    {
        return name;
    }
    
    /**
     * Gets number of instances in dataset
     * 
     * @return int - the number of instances in the dataset
     */
    public int numInstances()
    {
        if (data != null)
            return data.numInstances();
        else
            return 0;
    }

    /**
     * Gets number of attributes in dataset
     * 
     * @return int - the number of attributes in the dataset
     */
    public int numAttributes()
    {
        if (data != null)
            return data.numAttributes();
        else
            return 0;
    }
    
    /**
     * Gets the class index of the dataset
     * @return int - the class index 
     */
    public int classIndex()
    {
        if(data != null)
            return data.classIndex();
        else
            return -1;
    }
    
    /**
     * Gets the Instances object representing the dataset
     * @return the Instances object representing the dataset
     */
    public Instances getInstances()
    {
        return data;
    }

    public void setName(String name)
    {
        this.name = name;
    }
    
    /**
     * Sets the class index
     * 
     * @param classIndex The index of the class attribute
     */
    public void setClassIndex(int classIndex)
    {
        data.setClassIndex(classIndex);
    }
    
    /**
     * Saves data to a file with the given path
     * 
     * @throws  Exception
     */
    public boolean saveFile(String path)
    {
        try
        {
            DataSink.write(path, data);
            this.path = path;
            return true;
        }
        catch(Exception e)
        {
            return false;
        }
    }

    /**
     * Checks if the dataset has data loaded
     * 
     * @return boolean - true if the dataset has loaded data
     */
    public boolean hasData()
    {
        if (data == null)
        {
            return false;
        }

        return true;
    }
    
    /**
     * Adds an Instance to the dataset
     * 
     * @param instance the Instance to be added to the dataset
     */
    public void addInstance(Instance instance)
    {
        data.add(instance);
    }

    /**
     * Loads dataset from the provided file into the data Instances object.
     * 
     * @param path The path to the file containing the dataset.
     * @throws Exception
     */
    public void loadData(String path)
    {

        System.out.println("Loading data from " + path + "...");

        try
        {
            // See p.210 of the WEKA 3-8-2 manual
            data = DataSource.read(path);
            this.path = path;

        } catch (Exception e)
        {
            System.err.println("Error reading " + path);
            System.err.println(e.getMessage());
        }
    }

    /**
     * Loads dataset from a pre-existing Instances object.
     * 
     * @param instances The data to be copied
     */
    public void loadData(Instances instances)
    {
        data = new Instances(instances);
        path = null;
    }

    /**
     * Discretize data using WEKA implementation of Fayyad & Irani MDL
     * discretization. See WEKA 3-8-2 Manual p.219.
     * 
     * @param savePath Path to save discretized data to. The file extension (.arff or .csv) is specified when providing savePath.
     *
     * @return Returns the discretized data                 
     */
    public Instances discretize(String savePath)
    {
        
        // Check for class index being set
        if (data.classIndex() == -1)
        {
            System.err.println("Class index not set. Set class index prior to discretization.");
            return null;
        }

        Instances discretizedData = null;
        Discretize discretizer = new Discretize();

        try
        {
            // discretize data
            discretizer.setInputFormat(data);
            discretizedData = Filter.useFilter(data, discretizer);
            
            if(savePath != null)
            {
                DataSink.write(savePath, discretizedData);
            }
            
            return discretizedData;
        } 
        catch (Exception e)
        {
            System.out.println("Error discretizing data:");
            System.err.println(e.getMessage());
            return null;
        } 
    }
}
