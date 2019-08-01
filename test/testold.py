import os
import unittest
import numpy as np
#from src.pubmed import fetchByPubmed, fetchByQuery
from src.commons import wordsImportance, cleaningText, stoplist, createFrequencyModel, createListOfTextFromListOfFileNameByRow
from src.embedding import trainingModel, plotPCA, getAttributeVector, usableAttributeVector, computeSimilarity, completeSimilarityOfDatasets, getWordAggregationFromFile, cleaningDataset
from src.evalaluate import evaluation
from src.kgmanagement import getEntitiesPropertiesValue
from src.predefined import DATA_FOLDER


class TestLinkage(unittest.TestCase):
    # 0 corpus embedding
    # def test_corpusEmbedding(self):

    #     vectorSize = [100, 200, 300]
    #     windowSize = [2, 3, 5]
    #     for vec in vectorSize:
    #         for win in windowSize:
    #             for columnN in [LISTOFPROPERTIES, "description"]:
    #                 trainingModel(stoplist, dataSetFile,
    #                               columnN, 1, vec, win, 1)
    # 1 Create for each knowledge base file (ttl) it properties file.
    # For our case we have to call the src/kgmanagement/getEntitiesPropertiesValue funciton three times
    # - 1 for our first knowledge base file
    # - 2 for our second knowledge base file
    # - 3 for our ground truth knowledge base file
    #
    # def test_getEntitiesPropertiesValue(self):
    #     getEntitiesPropertiesValue(
    #         "oryzabase_testold.ttl", None, "Datasets")

    # 2 Create for each knowledge base file (ttl) it frequency model. This frequency model will be use as weights for words vectors
    def test_createFrequnecyModel(self):
        listOfKBCSVFile = []
        listOfKBCSVFileFolder = []
        listOfFqModel = []
        listOfFqModelFolder = []
        listOfAttributs = ['description', None]
        for attrib in listOfAttributs:
            folder, fileName = getEntitiesPropertiesValue(
                "oryzabase_testold.ttl", attrib, "Datasets")
            print("Dataset CSV file name ", fileName)
            print("Dataset CSV file folder ", folder)
            listOfKBCSVFile.append(fileName)
            listOfKBCSVFileFolder.append(folder)

            listOfFrequencies = ["tfidf"]
            #['tf', 'idf', 'tfidf']
            for freq in listOfFrequencies:
                fqModelfolder, fqModel = createFrequencyModel(
                    fileName, attrib, None, "row", "KB", freq, folder)
                print("Model file name ", fqModel)
                print("Model file folder ", folder)
                listOfFqModel.append(fqModel)
                listOfFqModelFolder.append(fqModelfolder)

        for kbcsvFile in range(len(listOfKBCSVFile)):
            for fqmodel in range(len(listOfFqModel)):
                for freq in listOfFrequencies:
                    filePath = os.path.join(os.path.join(
                        DATA_FOLDER, listOfKBCSVFileFolder[kbcsvFile]), listOfKBCSVFile[kbcsvFile])
                    if freq == "tfidf" and os.path.getsize(filePath) > 0:
                        wordsImpFileFolder, wordsImpFile = wordsImportance(
                            listOfFqModel[fqmodel], freq, listOfKBCSVFile[kbcsvFile], attrib, None, "row", listOfFqModelFolder[fqmodel], listOfKBCSVFileFolder[kbcsvFile])
                        print("Word important file name ", wordsImpFile)
                        print("Word important file folder ", wordsImpFileFolder)

    # 3 Create a file containing words and their tf or idf or tfidf.

    # def test_wordsImoprtance(self):
    #     fileNameTf = "KBrowtfCount20190710050637"
    #     fileNameIdf = "KBrowidfValue20190710164622"
    #     fileNameTfIdf = "KBrowtfIdfValue20190710050637"
    #     wordsImportance(fileNameTfIdf, "tfidf",
    #                     "gramene_Oryza_sativa_japonica_genes.csv", columnName="description")
    # wordsImportance(fileNameIdf, "idf",
    #                 "gramene_Oryza_sativa_japonica_genes.csv", columnName="description")

    # 4 This is an intermediate stage to have for a give attribute of an entity a dictionary with words as keys an value of these keys (words) their vector representations from the embedding model of corpus
    # def test_getAttributeVector(self):
    #     model = "myModel.bin"
    #     print(getAttributeVector(model, "gramene_Oryza_sativa_japonica_genes.csv",
    #                              "OS02G0461200", entityProperty="description", folder="Texts"))

    # 5 create usable vectors of an entity
    # def test_usableVector(self):
    #     model = "myModel.bin"
    #     fileNameTfIdf = "rowvocabularyTFIDFOf20190712114942.csv"
    #     vectorSize, attributeVector = getAttributeVector(model, "gramene_Oryza_sativa_japonica_genes.csv",
    #                                                      "OS02G0461200", entityProperty=["description", "label"], folder="Texts")
    #     final = usableAttributeVector(fileNameTfIdf, "tfidf",
    #                                   "OS02G0461200", attributeVector, vectorSize, folder="Outputs")
    #     print("\n\n#########################\n\n")
    #     print(final)

    # 6 compute similarity between two entities. The same model is used for both entities.
    # For the frequency file, each entity uses the file of it knowledge based

    # def test_similarity(self):
    #     model = "myModel.bin"
    #     fileNameTfIdf = "rowvocabularyTFIDFOf20190712114942.csv"
    #     vectorSize, attributeVector = getAttributeVector(model, "gramene_Oryza_sativa_japonica_genes.csv",
    #                                                      "OS02G0461200", entityProperty=["description", "label"], folder="Texts")
    #     vectorOne = usableAttributeVector(fileNameTfIdf, "tfidf",
    #                                       "OS02G0461200", attributeVector, vectorSize, folder="Outputs")

    #     print(computeSimilarity(vectorOne, vectorOne))
    # 7 compute complete similarity between two database files.
    # def test_completeSimilarity(self):
    #     corpusModel = "myModel.bin"
    #     model = "tfidf"
    #     fileNameTfIdfOne = "rowvocabularyTFIDFOf20190712114942.csv"
    #     fileNameTfIdfTwo = "rowvocabularyTFIDFOf20190712114942.csv"
    #     databaseOne = "gramene_Oryza_sativa_japonica_genes.csv"
    #     databaseTwo = "gramene_Oryza_sativa_japonica_genes.csv"
    #     completeSimilarityOfDatasets(
    #         corpusModel, model, databaseOne, fileNameTfIdfOne, databaseTwo, fileNameTfIdfTwo)


if __name__ == "__main__":
    myTest = TestLinkage()
    # myTest.test_wordsyImportance()

   # myTest.wordsImportance()
   # 1. Extract properties and their values from .ttl dataset
   # myTest.test_getEntitiesPropertiesValue()

   # 2. Get statistics of keywords of properties in text
   # 2.1 Get keywords from each property

   # 2.2 Get statistics for keywords

   # 3. Embed the text
   # myTest.test_embedding()

   # 4. Identify keywords vectors for for entities

   # 5. Aggregate keywords vectors to form entities vectors

   # myTest.test_FindElementInListOfList()
   # myTest.test_kMeans()
   # myTest.test_calculateCenter()
   # myTest.test_compareSelectedVectors()
   # myTest.test_completeKmeans()
