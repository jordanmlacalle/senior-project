package test;

import static org.junit.Assert.*;


import org.junit.Before;
import org.junit.Test;

import com.jordanml.TransactionClassifier.Dataset;

import weka.core.Instance;

/**
 * JUnit test class for the Dataset class. 100% coverage for Dataset.java.
 * 
 * @author Jordan
 *
 */
public class DatasetTest
{

    public static Dataset defaultDataset, nominalClass, numericalClass, testDataset;

    @Before
    public void iniTestDataset()
    {
        try
        {
            defaultDataset = new Dataset();
            testDataset = new Dataset();
            numericalClass = new Dataset("../data/creditcard.arff");
            nominalClass = new Dataset("../data/creditcard_nom.arff");
        }
        catch (Exception e)
        {
            System.out.println("Could not load data");
            System.err.println(e.getMessage());
        }
    }

    /**
     * Test default constructor instantiation
     */
    @Test
    public void testDataset()
    {

        try
        {
            new Dataset();
        } 
        catch (Exception e)
        {
            fail(e.getMessage());
        }
    }

    /**
     * Test instantiation with existing file as parameter
     */
    @Test
    public void testDatasetStringExistingFile()
    {

        try
        {
            new Dataset("../data/creditcard_nom.arff");
        } 
        catch (Exception e)
        {
            fail(e.getMessage());
        }
    }

    /**
     * Test instantiation with non-existent file
     */
    @Test
    public void testDatasetStringError()
    {
        try
        {
            new Dataset("");
        } 
        catch (Exception e)
        {

        }
    }

    /**
     * Test constructor w/ Instances object as parameter
     */
    @Test
    public void testDatasetInstances()
    {

        try
        {
            new Dataset(nominalClass.data);
        }
        catch (Exception e)
        {
            fail("Could not load data");
        }
    }

    /**
     * Test getters
     */
    @Test
    public void testGetPath()
    {
        assertEquals("Overridden constructor should initialize path to provided path", "../data/creditcard_nom.arff", nominalClass.getPath());
        assertEquals("Default constructor should initialize path to null", null, defaultDataset.getPath());
    }

    @Test
    public void testNumInstances()
    {

        assertEquals(284807, nominalClass.numInstances());
        assertEquals(0, defaultDataset.numInstances());
    }

    @Test
    public void testNumAttributes()
    {

        assertEquals(31, nominalClass.numAttributes());
        assertEquals(0, defaultDataset.numAttributes());
    }

    /**
     * Test whether the loadData method properly sets the values for instances,
     * attributes, path, and data.
     */
    @Test
    public void testLoadData()
    {

        try
        {
            testDataset.loadData("../data/creditcard_nom.arff");
        } catch (Exception e)
        {
            fail("Could not load data");
        }
        assertEquals(284807, testDataset.numInstances());
        assertEquals(31, testDataset.numAttributes());
        assertEquals("../data/creditcard_nom.arff", testDataset.getPath());
        assertEquals(true, testDataset.hasData());
    }

    /**
     * Test loading data from Instances object into empty Dataset
     */
    @Test
    public void testLoadDataInstances()
    {

        try
        {
            testDataset.loadData(nominalClass.data);
        } catch (Exception e)
        {
            fail("Could not load data");
        }
        assertEquals(284807, testDataset.numInstances());
        assertEquals(31, testDataset.numAttributes());
        assertEquals(null, testDataset.getPath());
        assertEquals(true, testDataset.hasData());

    }

    /**
     * Test checking Dataset to see if data exists
     */
    @Test
    public void testHasData()
    {

        assertEquals("Data should not be null", true, nominalClass.hasData());
        assertEquals("Data should be null", false, defaultDataset.hasData());
    }

    /**
     * Test discretization with dataset that has a nominal class and class index is
     * set
     */
    @Test
    public void testDiscretizeNominalClass()
    {
        nominalClass.data.setClassIndex(nominalClass.numAttributes() - 1);

        try
        {
            nominalClass.discretize("../data/test/testDiscretize.arff");
        } catch (Exception e)
        {
            fail(e.getMessage());
        }
    }

    /**
     * Test discretization with dataset that has a numerical class and class index
     * is set. Because discretization can only be performed with a nominal class, an
     * exception should be thrown.
     */
    @Test
    public void testDiscretizeNumericalClass()
    {
        numericalClass.data.setClassIndex(numericalClass.numAttributes() - 1);

        assertEquals("Should return null", null, numericalClass.discretize("../data/test/numericalClassShouldFail.arff"));
    }
    
    /**
     * Test discretization with dataset that does not have class index set (class
     * index == -1) Class index must be set for discretization.
     */
    @Test
    public void testDiscretizeNoClassIndex()
    {
        // did not set class index
        try
        {
            assertEquals("Should return null when class index is not set", null, numericalClass.discretize(null));
        } catch (Exception e)
        {
            fail("Should not reach exception");
        }

    }

    /**
     * Test adding an Instance object to the Instances data object in a Dataset
     */
    @Test
    public void testAddInstance()
    {

        int numInstances = nominalClass.numInstances();
        Instance instance = (Instance) nominalClass.data.instance(0).copy();

        nominalClass.addInstance(instance);
        assertEquals(numInstances + 1, nominalClass.numInstances());
    }

}
