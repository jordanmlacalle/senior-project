package test;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.*;
import com.jordanml.TransactionClassifier.Dataset;

/**
 * JUnit test class for the Dataset class. 100% coverage for Dataset.java.
 * @author Jordan
 *
 */
class DatasetTest {
    
    public static Dataset defaultDataset, nominalClass, numericalClass, testDataset;
    /**
     * initialized dataset should have:
     * instances
     * attributes: 
     * path: src/data/creditcard_nom.arff
     */
    
    @BeforeEach
    void iniTestDataset() {
        defaultDataset = new Dataset();
        testDataset = new Dataset();
        numericalClass = new Dataset("../data/creditcard.arff");
        nominalClass = new Dataset("../data/creditcard_nom.arff");
    }

    /**
     * Test default constructor instantiation
     */
    @Test
    void testDataset() {
        
        try {
            new Dataset();
        } catch (Exception e) {
            fail(e.getMessage());
        }
    }

    /**
     * Test instantiation with existing file as parameter
     */
    @Test
    void testDatasetStringExistingFile() {
        
        try {
            new Dataset("../data/creditcard_nom.arff");
        } catch (Exception e) {
            fail(e.getMessage());
        }
    }

    /**
     * Test instantiation with non-existent file
     */
    @Test
    void testDatasetStringError() {
        
        try {
            new Dataset("nonexistent.arff");
            fail("Expected exception due to nonexistent file");
        } catch (Exception e) {
            
        }
        try {
            new Dataset("");
            fail("Expected exception due to empty path");
        } catch (Exception e) {
            
        }
    }
    
    /**
     * Test path getter
     */
    @Test
    void testGetPath() {
        
        assertEquals("../data/creditcard_nom.arff", nominalClass.getPath(), "Overridden constructor should initialize path to provided path");
        assertEquals(null, defaultDataset.getPath(), "Default constructor should initialize path to null");
    }

    @Test
    void testNumInstances() {
        
        assertEquals(284807, nominalClass.numInstances(), "Overloaded constructor should initialize instances to the number of instances in the dataset");
        assertEquals(0, defaultDataset.numInstances(), "Default constructor should initialize instances to 0");
    }

    @Test
    void testNumAttributes() {
        
        assertEquals(31, nominalClass.numAttributes(), "Overloaded constructor should initialize attributes to the number of attributes in the dataset");
        assertEquals(0, defaultDataset.numAttributes(), "Default constructor should initialize attributes to 0");
    }

    /**
     * Test whether the loadData method properly sets the values for instances, attributes, path, and data.
     */
    @Test
    void testLoadData() {
        
        testDataset.loadData("../data/creditcard_nom.arff");
        assertEquals(284807, testDataset.numInstances());
        assertEquals(31, testDataset.numAttributes());
        assertEquals("../data/creditcard_nom.arff", testDataset.getPath());
        assertEquals(true, testDataset.hasData());
    }

    @Test
    void testHasData() {
        
        assertEquals(true, nominalClass.hasData(), "data should not be null");
        assertEquals(false, defaultDataset.hasData(), "data should be null");
    }
    
    /**
     * Test discretization with dataset that has a nominal class and class index is set
     */
    @Test
    void testDiscretizeNominalClass() {
        nominalClass.data.setClassIndex(nominalClass.numAttributes()-1);
        
        try {
            nominalClass.discretize("../data/test/testDiscretize.arff");
        } catch (Exception e) {
            fail(e.getMessage());
        }
    }
    
    /**
     * Test discretization with dataset that has a numerical class and class index is set.
     * Because discretization can only be performed with a nominal class, an exception should be thrown.
     */
    @Test
    void testDiscretizeNumericalClass() {
        
        numericalClass.data.setClassIndex(numericalClass.numAttributes()-1);
       
        try {
            numericalClass.discretize("../data/test/numericalClassShouldFail.arff");
            fail("Expected exception");
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }
    
    /**
     * Test discretization with dataset that has a nominal class and class index is set.
     * In this test, the call to the discretize method uses null as the file path argument.
     * Prior to discretization, path should be set to the path to the file used as the dataset souce.
     * After discretization, path should be set to null because the dataset was discretized and no path
     * was specified in order to save the file.
     */
    @Test
    void testDiscretizeReplaceData() {
        
        nominalClass.data.setClassIndex(nominalClass.numAttributes() - 1);
        
        try {
            assertEquals("../data/creditcard_nom.arff", nominalClass.getPath(), "Path should match set path");
            nominalClass.discretize(null);
            assertEquals(null, nominalClass.getPath(), "Path should be null after replacing data");
        } catch (Exception e) {
            fail("Should not reach exception");
        }
        
    }
    
    /**
     * Test discretization with dataset that does not have class index set (class index == -1)
     * Class index must be set for discretization.
     */
    @Test
    void testDiscretizeNoClassIndex() {
        //did not set class index
        try {
            assertEquals(false, numericalClass.discretize(null), "Should return false when class index is not set");
        } catch (Exception e) {
            fail("Should not reach exception");
        }
        
    }
    

}
