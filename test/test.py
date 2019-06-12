import os
import unittest
import numpy as np 
from src.commons import findElementInListOfList, kMeans, calculateCenter, compareSelectedVectors, completeKmeans, createCooccurrenceMatrix, createListOfText, generateTermDocumentMatrix, createTfIdfAndBowModel, stoplist, englishStopWords, createCommonVocabulary, MODEL
from src.kgmanagement import getEntitiesPropertiesValue, LISTOFPROPERTIES
from src.embedding import trainingModel, plotPCA, getAttributeVector, getDicOfAttributesVectors

class TestLinkage(unittest.TestCase):
    def test_getDicOfAttributesVectors(self):
        getDicOfAttributesVectors("OryzabaseGeneList_test.csv")
    
    def test_getAttributeVector (self):
        getAttributeVector("myModel.bin", "OryzabaseGeneList_test_name.csv")

        
    def test_createCommonVocabulary(self):
        listOfFiles = ["oryzabase_test.csv","OryzabaseGeneList_test.csv"]
        Vocab, VocabSize =createCommonVocabulary(stoplist, listOfFiles, LISTOFPROPERTIES, "Outputs" )
        return Vocab, VocabSize
        
        
    def test_FindElementInListOfList(self):
        L = [[3,4,6],[1,8],[3,7,9]]
        e = 9
        self.assertTrue(findElementInListOfList(L,e))

        
    def test_kMeans(self):
        listOfVectors = [[1,1], [5,3], [2,1], [4,3], [5,4],[4,4]]
        k = 2
        kmeans = kMeans(listOfVectors, k)
        if k >= 6:
            self.assertEqual(kmeans, [])
        else:
            self.assertTrue(kmeans[0] == [0,2] and kmeans[1] == [1,3,4,5], 'Bad classification')
        
    def test_calculateCenter(self):
        listOfVectors = [[1,1], [5,3], [2,1], [4,3], [5,4],[4,4]]
        listOfPoints = [1,3,4,5]
        self.assertTrue(np.all(calculateCenter(listOfPoints, listOfVectors) == [(5+4+5+4)/4,(3+3+4+4)/4]))


    def test_getEntitiesPropertiesValue(self):
        """
        Our Knowledge dataset:
        1. oryzabase.ttl
        2. OryzabaseGeneList.ttl 
       
        This test is suppose to create the CSV file of the entry 
        file with colunms representing the properties of     interest 
        of each entity of the KB.
        """
        fileName = "oryzabase_test.ttl"
        getEntitiesPropertiesValue(fileName)


    def test_compareSelectedVectors(self):
        v1 = [[0,2], [1,3,4]]
        v2 = [[0,2],[1,3],[4]]
        self.assertFalse(compareSelectedVectors(v1,v2))
        v3 = [[0,2], [1,3,4]]
        v4 = [[0,2],[1,4,3]]
        self.assertTrue(compareSelectedVectors(v3,v4))


    def test_completeKmeans(self):
        listOfVectors = [[1,2], [2,1], [2,4], [4,2], [4,3], [5,3], [5,2], [2,3], [1,3], [1,1], [3,1], [3,3]]
#=[[1,1],[2,2],[2,3],[3,1],[4,1],[4,2],[5,2],[5,3],[5,5],[4,5],[6,2],[6,3],[6,4]]
        #[[1,1],[5,3], [2,1], [4,3], [5,4],[4,4]]
        k = 4
        #v4 = [[0,2], [1,3,4]]
        l = completeKmeans([], listOfVectors, k, 9)
        #self.assertTrue(compareSelectedVectors(l, v4))


    def create_createCooccurrenceMatrix(self):
        listOfText = createListOfText("gramene_Oryza_sativa_japonica_genes_test2.csv", "description" )
        c = createCooccurrenceMatrix(stoplist, listOfText)
        print(c)


    def test_integration(self):
        k = 6
        itterations = 9
        listOfText = createListOfText("gramene_Oryza_sativa_japonica_genes_test2.csv", "description" )
        
        cooccurrenceMat = createCooccurrenceMatrix(stoplist, listOfText)
        l  = completeKmeans([], cooccurrenceMat, k, itterations)
        print(l)


    def test_createTfIdfAndBowModel(self):
        listOfText = createListOfText("gramene_Oryza_sativa_japonica_genes.csv", "description" )
        createTfIdfAndBowModel()

    def test_generateTermDocumentMatrix(sefl):
        listOfText = createListOfText("gramene_Oryza_sativa_japonica_genes.csv", "description" )
        X = generateTermDocumentMatrix(listOfText)
        return X

    def test_stoplist(self):
        print("Stop words ", stoplist)

    def test_embedding(self):
        dataSetFile = "data_text.csv"
        trainingModel(stoplist, dataSetFile, "Abstract")

    def test_plotPCA(self):
        myModel = os.path.join(MODEL, "myModel.bin")
        plotPCA(myModel, 2)
        
if __name__=="__main__":
    myTest = TestLinkage()
    
    # 1. Extract properties and their values from .ttl dataset
    myTest.test_getEntitiesPropertiesValue()
    
    # 2. Get statistics of keywords of properties in text
    # 2.1 Get keywords from each property
    
    # 2.2 Get statistics for keywords
    
    # 3. Embed the text
    myTest.test_embedding()
    
    # 4. Identify keywords vectors for for entities

    # 5. Aggregate keywords vectors to form entities vectors

    #myTest.test_FindElementInListOfList()
    #myTest.test_kMeans()
    #myTest.test_calculateCenter()
    #myTest.test_compareSelectedVectors()
    #myTest.test_completeKmeans()
