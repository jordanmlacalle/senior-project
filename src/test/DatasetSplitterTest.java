/**
 * 
 */
package test;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Before;
import org.junit.Test;

import com.jordanml.TransactionClassifier.Dataset;
import com.jordanml.TransactionClassifier.DatasetSplitter;

import weka.core.Instances;

/**
 * @author Jordan
 *
 */
public class DatasetSplitterTest
{

    static String testSource;
    static Dataset testDataset;
    static DatasetSplitter testSplitter;

    /**
     * @throws java.lang.Exception
     */
    @Before
    public void setUpBeforeClass() throws Exception
    {
        testSource = "../data/breast-cancer.arff";
        testDataset = new Dataset(testSource);
        testSplitter = new DatasetSplitter(testDataset.getInstances());
    }

    /**
     * Test method for
     * {@link com.jordanml.TransactionClassifier.DatasetSplitter#DatasetSplitter(com.jordanml.TransactionClassifier.Dataset)}.
     */
    @Test
    public void testDatasetSplitter()
    {
        testSplitter = new DatasetSplitter(testDataset.getInstances());
        assertEquals(0, testSplitter.getFolds().size());
        assertEquals(testDataset.numInstances(), testSplitter.getSource().numInstances());
    }

    /**
     * Test that the correct number of folds are allocated
     */
    @Test
    public void testInitFolds()
    {

        int numFolds = 10;

        testSplitter.initFolds(numFolds);
        assertEquals(numFolds, testSplitter.getFolds().size());
    }

    /**
     * Test folds getter
     */
    @Test
    public void testGetFolds()
    {

        ArrayList<Dataset> testFolds;
        int numFolds = 10;

        testSplitter.initFolds(numFolds);
        testFolds = testSplitter.getFolds();
        assertEquals(numFolds, testFolds.size());
    }

    /**
     * Test method checking if folds have been initialized
     */
    @Test
    public void testIsInitialized()
    {

        int numFolds = 10;

        assertEquals(false, testSplitter.isInitialized());
        testSplitter.initFolds(numFolds);
        assertEquals(true, testSplitter.isInitialized());
    }
    
    @Test
    public void testSplitData()
    {
        int numFolds = 10;
        
        //Folds not initialized, should return false
        assertEquals(false, testSplitter.splitData());
        testSplitter.initFolds(numFolds);
        //Folds initialized, should return true once complete
        assertEquals(true, testSplitter.splitData());
    }
    
    @Test
    public void getSource()
    {
        Instances source = testSplitter.getSource();
        assertNotNull(source);
        assertEquals(testDataset.numAttributes(), source.numAttributes());
        assertEquals(testDataset.numInstances(), source.numInstances());
    }
}
