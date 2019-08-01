import os
import numpy as np
#from src.pubmed import fetchByPubmed, fetchByQuery
from src.commons import wordsImportance, cleaningText, stoplist, createFrequencyModel, createListOfTextFromListOfFileNameByRow
from src.embedding import trainingModel, plotPCA, getAttributeVector, usableAttributeVector, computeSimilarity, completeSimilarityOfDatasets, getWordAggregationFromFile, cleaningDataset
from src.evalaluate import evaluation
from src.kgmanagement import getEntitiesPropertiesValue
from src.predefined import LISTOFPROPERTIES

# 0 corpus embedding


# def corpusEmbedding():
#     dataSetFile = "newdata.csv"
#     columnN = "Abstract"
#     vectorSize = [100, 200, 300]
#     windowSize = [2, 3, 5]
#     for vec in vectorSize:
#         for win in windowSize:
#             trainingModel(stoplist, dataSetFile,
#                           columnN, 1, vec, win, 1, 1, None, "row", "Texts")
# 1 Create for each knowledge base file (ttl) it properties file.
# For our case we have to call the src/kgmanagement/getEntitiesPropertiesValue funciton three times
# - 1 for our first knowledge base file
# - 2 for our second knowledge base file
# - 3 for our ground truth knowledge base file
#
# 1.1
# def getEntitiesPropertiesValue(self):
#     getEntitiesPropertiesValue(
#         "oryzabase_testold.ttl")

# 2 Create for each knowledge base file (ttl) it frequency model. This frequency model will be use as weights for words vectors


def createTheFrequencyModel():
    listOfKBCSVFile = []
    listOfKBCSVFileFolder = []
    listOfModelUsed = []
    listOfKB = ["gramene_Oryza_sativa_japonica_genes.ttl", "oryzabase.ttl"]
    listOfAttributs = ['description', None]
    for KB in listOfKB:
        for attrib in listOfAttributs:
            kbcsvFilefolder, kbcsvFile = getEntitiesPropertiesValue(
                KB, attrib, "Datasets", "Outputs")
            listOfKBCSVFile.append(kbcsvFile)
            listOfKBCSVFileFolder.append(kbcsvFilefolder)
    listOfFrequencies = ['tf', 'idf', 'tfidf']
    listOfFqModel = []
    listOfFqModelFolder = []
    for kbcsvFile in range(len(listOfKBCSVFile)):
        for freq in listOfFrequencies:
            for attrib in listOfAttributs:
                if freq == "tfidf":
                    fqModelfolder, fqModel = createFrequencyModel(
                        listOfKBCSVFile[kbcsvFile], attrib, "row", "KB", freq, listOfKBCSVFileFolder[kbcsvFile])
                    listOfFqModel.append(fqModel)
                    listOfFqModelFolder.append(fqModelfolder)

    for kbcsvFile in range(len(listOfKBCSVFile)):
        for fqmodel in range(len(listOfFqModel)):
            for freq in listOfFrequencies:
                if freq == "tfidf":
                    wordsImpFileFolder, wordsImpFile = wordsImportance(
                        listOfFqModel[fqmodel], freq, listOfKBCSVFile[kbcsvFile], attrib, "row", listOfFqModelFolder[fqmodel], listOfKBCSVFileFolder[kbcsvFile])
    corpusModels = []
    model = "tfidf"
    for corpusModel in corpusModels:
        completeSimilarityOfDatasets(
            corpusModel, model, listOfKBCSVFile[0], listOfKBCSVFileFolder[0], listOfKBCSVFile[1], listOfKBCSVFileFolder[1], attrib, "Models", "Texts", "Outputs")

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
    # corpusEmbedding()
    createTheFrequencyModel()
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
