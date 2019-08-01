import os
import traceback
import unittest
from datetime import datetime
import numpy as np
#from src.pubmed import fetchByPubmed, fetchByQuery
from src.commons import wordsImportance, cleaningText, stoplist, createFrequencyModel, createListOfTextFromListOfFileNameByRow
from src.embedding import trainingModel, plotPCA, getAttributeVector, usableAttributeVector, computeSimilarity, completeSimilarityOfDatasets, getWordAggregationFromFile, cleaningDataset
from src.evalaluate import evaluation
from src.kgmanagement import getEntitiesPropertiesValue
from src.predefined import DATA_FOLDER, LISTOFPROPERTIES


# 0 corpus embedding
def test_corpusEmbedding():
    listOfModel = []
    listOfModelFolder = []
    dataSetFile = "newdata.csv"
    t = stoplist
    vectorSize = [25, 50, 100, 150, 200, 250, 300]
    windowSize = [2, 3, 5]
    for vec in vectorSize:
        for win in windowSize:
            folder, model = trainingModel(
                t, dataSetFile, "Abstract", 1, vec, win, 1, 3, None, "row", "Texts")
            listOfModel.append(model)
            listOfModelFolder.append(folder)
    return listOfModelFolder, listOfModel

# 1 Create for each knowledge base file (ttl) it properties file.
# For our case we have to call the src/kgmanagement/getEntitiesPropertiesValue funciton three times
# - 1 for our first knowledge base file
# - 2 for our second knowledge base file
# - 3 for our ground truth knowledge base file


def test_getAttributeVectorGround():
    getEntitiesPropertiesValue(
        "oryzabase_ground.ttl", None, "Grounds", "Outputs")
    print("End ground")

# def test_getEntitiesPropertiesValue(self):
#     getEntitiesPropertiesValue(
#         "oryzabase_testold.ttl", None, "Datasets")

# 2 Create for each knowledge base file (ttl) it frequency model. This frequency model will be use as weights for words vectors


def test_final(log):
    # gFileFolder, gFile = getEntitiesPropertiesValue(
    #     "oryzabase_ground.ttl", None, "Grounds", "Outputs")

    listOfCorpusModel = ["Word2VecModel_Skipgram_Abstract_win_5_vec_300_20190731164233.bin", "Word2VecModel_Skipgram_Abstract_win_3_vec_300_20190731164123.bin", "Word2VecModel_Skipgram_Abstract_win_2_vec_300_20190731164016.bin", "Word2VecModel_Skipgram_Abstract_win_5_vec_250_20190731163922.bin", "Word2VecModel_Skipgram_Abstract_win_3_vec_250_20190731163808.bin", "Word2VecModel_Skipgram_Abstract_win_2_vec_250_20190731163653.bin", "Word2VecModel_Skipgram_Abstract_win_5_vec_200_20190731163544.bin", "Word2VecModel_Skipgram_Abstract_win_3_vec_200_20190731163428.bin", "Word2VecModel_Skipgram_Abstract_win_2_vec_200_20190731163319.bin", "Word2VecModel_Skipgram_Abstract_win_5_vec_150_20190731163215.bin", "Word2VecModel_Skipgram_Abstract_win_3_vec_150_20190731163119.bin", "Word2VecModel_Skipgram_Abstract_win_2_vec_150_20190731163040.bin", "Word2VecModel_Skipgram_Abstract_win_5_vec_100_20190731163005.bin", "Word2VecModel_Skipgram_Abstract_win_3_vec_100_20190731162928.bin", "Word2VecModel_Skipgram_Abstract_win_2_vec_100_20190731162853.bin", "Word2VecModel_Skipgram_Abstract_win_5_vec_50_20190731162821.bin", "Word2VecModel_Skipgram_Abstract_win_3_vec_50_20190731162747.bin", "Word2VecModel_Skipgram_Abstract_win_2_vec_50_20190731162715.bin", "Word2VecModel_Skipgram_Abstract_win_5_vec_25_20190731162354.bin", "Word2VecModel_Skipgram_Abstract_win_3_vec_25_20190731162300.bin", "Word2VecModel_Skipgram_Abstract_win_2_vec_25_20190731162209.bin",
                         ]

    #listOfCorpusModelFolder, listOfCorpusModel = test_corpusEmbedding()

    # getEntitiesPropertiesValue(
    #     "oryzabase_ground.ttl", None, "Grounds", "Outputs")
    # print("End ground")
    try:
        listOfKBCSVFile = []
        listOfKBCSVFileFolder = []

        listOfFqModel = []
        listOfFqModelFolder = []

        listOfWordsImportance = []
        listOfWordsImportanceFolder = []

        listOfAttributs = ['description', None]  # ok

        for attrib in listOfAttributs:
            for kb in ["gramene_Oryza_sativa_japonica_genes.ttl", "oryzabase.ttl"]:
                folder, fileName = getEntitiesPropertiesValue(
                    kb, attrib, "Datasets")
                print("Dataset CSV file name ", fileName)
                print("Dataset CSV file folder ", folder)
                listOfKBCSVFile.append(fileName)
                listOfKBCSVFileFolder.append(folder)

                fqModelfolder, fqModel = createFrequencyModel(
                    fileName, attrib, None, "row", "KB", "tfidf", folder)
                print("Model file name ", fqModel)
                print("Model file folder ", folder)
                listOfFqModel.append(fqModel)
                listOfFqModelFolder.append(fqModelfolder)

                filePath = os.path.join(os.path.join(
                    DATA_FOLDER, folder), fileName)
                wordsImpFileFolder, wordsImpFile = wordsImportance(
                    fqModel, "tfidf", fileName, attrib, None, "row", fqModelfolder, folder)
                listOfWordsImportance.append(wordsImpFile)
                listOfWordsImportanceFolder.append(wordsImpFileFolder)
        crossDistanceList = []
        crossDistanceFolderList = []
        print("### listOfCorpusModel")
        print(listOfCorpusModel)
        print("###")
        print("### listOfKBCSVFile")
        print(listOfKBCSVFile)
        print("###")
        print("### listOfWordsImportance")
        print(listOfWordsImportance)
        print("###")
        print("### listOfWordsImportanceFolder")
        print(listOfWordsImportanceFolder)
        print("###")
        for indexmodel in range(len(listOfCorpusModel)):
            for attrib in listOfAttributs:
                crossDistanceFolder, crossDistance = completeSimilarityOfDatasets(listOfCorpusModel[indexmodel], "tfidf", listOfKBCSVFile[0], listOfWordsImportance[0], listOfKBCSVFile[1],
                                                                                  listOfWordsImportance[1], attrib, "Models", listOfKBCSVFileFolder[0], listOfWordsImportanceFolder[0])
                crossDistanceList.append(crossDistance)
                crossDistanceFolderList.append(crossDistanceFolder)
        gFile = "oryzabase_ground_Propertiesdescription-entity-explanation-has_alternative_name-has_rap_identifier-has_synonym-has_tigr_identifier-has_trait_class-has_uniprot_accession-label-name_20190730184911.csv"
        gFileFolder = "Outputs"
        test_evaluation(gFile, crossDistanceList,
                        gFileFolder, crossDistanceFolderList)
    except Exception:
        traceback.print_exc(file=log)


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

# 8 evaluation of the method
def test_evaluation(gFile, listOfrFile, gFileFolder, listOfrFileFolder):
    gColumnName = ["entity", "has_rap_identifier"]

    #listOfrFile = ["distancesCrossSimilaritydescription_tfidf_20190727174549.csv"]
    rColumnName = ["orizabase_B", "orizabase_A"]
    threshold = [value*0.1 for value in range(10, 50, 5)]
    for indexrFile in range(len(listOfrFile)):
        evaluation(gFile, gColumnName, listOfrFile[indexrFile], rColumnName,
                   threshold, 1, gFileFolder, listOfrFileFolder[indexrFile], True)


if __name__ == "__main__":
    log = open("Log_"+str(
        datetime.now()).replace(":", "").replace("-", "").replace(" ", "").split(".")[0] + ".txt", "a+")

    # test_corpusEmbedding()
    test_final(log)
    # test_createFrequnecyModel()
    #myTest = TestLinkage()
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
