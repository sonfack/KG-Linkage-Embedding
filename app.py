import os
import traceback
import unittest
from datetime import datetime
from multiprocessing import Process
import numpy as np
#from src.pubmed import fetchByPubmed, fetchByQuery
from src.baseline import getListOfText, createDistance
from src.commons import wordsImportance, cleaningText, stoplist, createFrequencyModel, createListOfTextFromListOfFileNameByRow, partitionDataset, readDataFile
from src.embedding import trainingModel, plotPCA, getAttributeVector, usableAttributeVector, computeSimilarity, completeSimilarityOfDatasets, getWordAggregationFromFile, cleaningDataset
from src.evalaluate import evaluation
from src.kgmanagement import getEntitiesPropertiesValue
from src.predefined import DATA_FOLDER, LISTOFPROPERTIES


# 0 corpus embedding

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
    #defPartions = [10, 20, 30, 50, 80, 100]

    listOfCorpusModel = [
        "Word2VecModel_Skipgram_Abstract_win_2_vec_200_20190731163240.bin"]


   #listOfCorpusModelFolder, listOfCorpusModel = test_corpusEmbedding()

   # getEntitiesPropertiesValue(
   #     "oryzabase_ground.ttl", None, "Grounds", "Outputs")
   # print("End ground")
    try:
        print("# OSJG")
        OSJGListOfKBCSVFileFolder = ["Outputs"]
        OSJGListOfKBCSVFile = [
            "gramene_Oryza_sativa_japonica_genes_Propertiesdescription-entity_20190731191854.csv"]
        OSJGListOfFqModel = [
            "KBrowtfIdfValue_Properties_description_tfidf_20190801063531"]
        # "KBrowtfIdfValue_Properties_description-entity-explanation-has_alternative_name-has_rap_identifier-has_synonym-has_tigr_identifier-has_trait_class-has_uniprot_accession-label-name_tfidf_20190803031757"
        OSJGListOfFqModelFolder = ["Models"]

        OSJGListOfWordsImportance = [
            "WordImportancerowvocabularyTFIDFOftfidf20190801063532.csv"]
        #    "WordImportancerowvocabularyTFIDFOftfidf20190803031758.csv"
        OSJGListOfWordsImportanceFolder = ["Outputs"]

        print("# O")
        OListOfKBCSVFileFolder = ["Outputs"]
        OListOfKBCSVFile = [
            "oryzabase_Propertiesdescription-entity_20190801164505.csv"]
        OListOfFqModel = [
            "KBrowtfIdfValue_Properties_description_tfidf_20190801213128"]
        # "KBrowtfIdfValue_Properties_description-entity-entity-explanation-has_alternative_name-has_rap_identifier-has_synonym-has_tigr_identifier-has_trait_class-has_uniprot_accession-label-name_tfidf_20190804153859"
        OListOfFqModelFolder = ["Models"]

        OListOfWordsImportance = [
            "WordImportancerowvocabularyTFIDFOftfidf20190801213128.csv"]
        #    "WordImportancerowvocabularyTFIDFOftfidf20190804153900.csv"
        OListOfWordsImportanceFolder = ["Outputs"]

        attrib = 'description'

        crossDistanceList = []
        crossDistanceFolderList = []

        gFile = "oryzabase_ground_Propertiesdescription-entity-explanation-has_alternative_name-has_rap_identifier-has_synonym-has_tigr_identifier-has_trait_class-has_uniprot_accession-label-name_20190730184911.csv"
        gFileFolder = "Outputs"

        listOfGFile = ["oryzabase_ground_Propertiesdescription-entity-explanation-has_alternative_name-has_rap_identifier-has_synonym-has_tigr_identifier-has_trait_class-has_uniprot_accession-label-name_20190730184911.csv"]
        listOfGFileFolder = ["Outputs"]
        """
        for indexmodel in range(len(listOfCorpusModel)):
            for part in range(len(defPartions)):
        """

        crossDistanceFolder, crossDistance = completeSimilarityOfDatasets(listOfCorpusModel[0], "tfidf", OSJGListOfKBCSVFile[0], OSJGListOfWordsImportance[
                                                                          0], OListOfKBCSVFile[0], OListOfWordsImportance[0], attrib, "Models", OSJGListOfKBCSVFileFolder[0], OListOfWordsImportanceFolder[0])
        crossDistanceList.append(crossDistance)
        crossDistanceFolderList.append(crossDistanceFolder)
        print("### evaluation")
        test_evaluation(listOfGFile[0], crossDistanceList,
                        listOfGFileFolder[0], crossDistanceFolderList)
        print("### end evaluation")
    except Exception:
        traceback.print_exc(file=log)


# 8 evaluation of the method
# def test_evaluation(gFile, listOfrFile, gFileFolder, listOfrFileFolder):

def test_evaluation(log):
    gColumnName = ["entity", "has_rap_identifier"]
    gFileFolder = "Outputs"
    gFile = "oryzabase_ground_Propertiesdescription-entity-explanation-has_alternative_name-has_rap_identifier-has_synonym-has_tigr_identifier-has_trait_class-has_uniprot_accession-label-name_20190730184911.csv"

    listOfrFile = "data/Outputs/distancesCrossSimilarity_PCA__BOW_20191022001118.csv"
    rColumnName = ["orizabase_B", "orizabase_A"]
    threshold = [value*0.1 for value in range(0, 10, 1)]
    try:
        evaluation(gFile, gColumnName, listOfrFile, rColumnName,
                   threshold, 2, gFileFolder, "Outputs", True)
    except Exception:
        traceback.print_exc(file=log)


if __name__ == "__main__":

    log = open("Log_eval"+str(
        datetime.now()).replace(":", "").replace("-", "").replace(" ", "").split(".")[0] + ".txt", "a+")

    test_evaluation(log)
