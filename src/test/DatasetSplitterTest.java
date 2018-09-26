/**
 * 
 */
package test;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import com.jordanml.TransactionClassifier.Dataset;
import com.jordanml.TransactionClassifier.DatasetSplitter;

/**
 * @author Jordan
 *
 */
public class DatasetSplitterTest {

    static String testSource;
    static Dataset testDataset;
    static DatasetSplitter testSplitter;
    
    /**
     * @throws java.lang.Exception
     */
    @Before
    public void setUpBeforeClass() throws Exception {
        testSource = "../data/breast-cancer.arff";
        testDataset = new Dataset(testSource);
        testSplitter = new DatasetSplitter(testDataset);
    }

    /**
     * Test method for {@link com.jordanml.TransactionClassifier.DatasetSplitter#DatasetSplitter(com.jordanml.TransactionClassifier.Dataset)}.
     */
    @Test
    public void testDatasetSplitter() {
        
        testSplitter = new DatasetSplitter(testDataset);
        
        assertEquals(0, testSplitter.getFolds().size(), "Expected size of folds ArrayList to be 0 after constructor call");
        assertEquals(testDataset.numInstances(), testSplitter.getSource().numInstances(), "Expected sourceData.numInstances() to be equivalent to testDataset.numInstances()");        
    }

    /**
     * Test that the correct number of folds are allocated 
     */
    @Test
    public void testInitFolds() {
        
        int numFolds = 10;
        
        testSplitter.initFolds(numFolds);
        assertEquals(numFolds, testSplitter.getFolds().size());
    }

    /**
     * Test folds getter
     */
    @Test
    public void testGetFolds() {
        
        ArrayList<Dataset> testFolds;
        int numFolds = 10;
        
        testSplitter.initFolds(numFolds);
        testFolds = testSplitter.getFolds();
        assertEquals(numFolds, testFolds.size(), "testFolds.size() does not match the expected number");
    }
    
    /**
     * Test method checking if folds have been initialized
     */
    @Test
    public void testIsInitialized() {
        
        int numFolds = 10;
        
        assertEquals(false, testSplitter.isInitialized(), "Uninitialized, isInitialized() should return false");
        testSplitter.initFolds(numFolds);
        assertEquals(true, testSplitter.isInitialized(), "Initialized, isInitialized() should return true");
    }

}
