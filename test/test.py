
import os
import unittest
import numpy as np
#from src.pubmed import fetchByPubmed, fetchByQuery
from src.commons import wordsImportance, cleaningText, stoplist, createFrequencyModel, createListOfTextFromListOfFileNameByRow, createListOfText, dropRowsWithEmptyProperty
from src.embedding import trainingModel, plotPCA, getAttributeVector, usableAttributeVector, computeSimilarity, completeSimilarityOfDatasets, getWordAggregationFromFile, cleaningDataset, retunModelVocabulary
from src.evalaluate import evaluation, analysisValues, numberOfImpEntity, returnDatasetVocabulary
from src.kgmanagement import getEntitiesPropertiesValue
from src.predefined import DATA_FOLDER
from src.analysis import plotAnalysis, plotWordPerKB


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
    def test_getAttributeVectorRDF(self):
        getEntitiesPropertiesValue(
            "DB_Lake.db2013.rdf", None, "Datasets", "Outputs")
        print("End RDF")

    def test_getAttributeVectorGround(self):
        getEntitiesPropertiesValue(
            "oryzabase_ground.ttl", None, "Grounds", "Outputs")
        print("End ground")

    def test_createFrequnecyModel(self):
        getEntitiesPropertiesValue(
            "oryzabase_ground.ttl", None, "Grounds", "Outputs")
        print("End ground")
        exit()
        listOfKBCSVFile = []
        listOfKBCSVFileFolder = []

        listOfFqModel = []
        listOfFqModelFolder = []

        listOfWordsImportance = []
        listOfWordsImportanceFolder = []
        listOfAttributs = ['description', None]
        for attrib in listOfAttributs:
            for kb in ["oryzabase_testold.ttl", "oryzabase_testold.ttl"]:
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
                #print("Word important file name ", wordsImpFile)
                #print("Word important file folder ", wordsImpFileFolder)
                listOfWordsImportance.append(wordsImpFile)
                listOfWordsImportanceFolder.append(wordsImpFileFolder)
        for attrib in listOfAttributs:
            completSimilarityFolder, completeSimilarityFile = completeSimilarityOfDatasets("myModel.bin", "tfidf", listOfKBCSVFile[0], listOfWordsImportance[0], listOfKBCSVFile[1],
                                                                                           listOfWordsImportance[1], attrib,  "Models", listOfKBCSVFileFolder[0], listOfWordsImportanceFolder[0])

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
    #
    # 8 evaluation of the method
    def test_evaluation(self):
        gFile = "oryzabase_ground_Propertiesdescription-entity-explanation-has_alternative_name-has_rap_identifier-has_synonym-has_tigr_identifier-has_trait_class-has_uniprot_accession-label-name_20190727202230.csv"
        gColumnName = ["entity", "has_rap_identifier"]

        rFile = "distancesCrossSimilaritydescription_tfidf_20190727174549.csv"
        rColumnName = ["orizabase_B", "orizabase_A"]
        threshold = [value*0.1 for value in range(10, 50, 5)]
        evaluation(gFile, gColumnName, rFile, rColumnName,
                   threshold, 1, "Outputs", "Outputs", True)

    # plot some database analysis
    def test_plot(self):
        numberOfEntityGOSJ, missingDescriptionGOSJ = analysisValues(
            "gramene_Oryza_sativa_japonica_genes_Propertiesdescription-entity_20190731191854.csv", "Outputs")

        numberOfEntityO, missingDescriptionO = analysisValues(
            "oryzabase_Propertiesdescription-entity_20190801164505.csv", "Outputs")

        plotAnalysis([numberOfEntityGOSJ, missingDescriptionGOSJ, numberOfEntityGOSJ-missingDescriptionGOSJ], [numberOfEntityO, missingDescriptionO, numberOfEntityO-missingDescriptionO], [
                     15772, 15772, 15772], "KG_GOSJG", "KG_O", "Ground truth", ('entities', 'Missing descriptions', 'Used entities'),  "Knowledge graphs", "Number of entities", " ")

    # analysis values to plot

    def test_analysis(self):
        analysisValues(
            "gramene_Oryza_sativa_japonica_genes_Propertiesdescription-entity_20190731191854.csv", "Outputs")

    #

    def test_entityPerKB(self):
        numberOfImpEntity(
            "WordImportancerowvocabularyTFIDFOftfidf20190803031758.csv", "Outputs")

    def test_wordsPerKB(self):
        # entities, descriptions = createListOfText("WordImportancerowvocabularyTFIDFOftfidf20190801063532.csv",
        #                                           "word", None, "row", "Outputs")
        # l = set()

        # for desc in range(len(descriptions)):
        #     l.add(descriptions[desc])
        # lKB_A = len(l)
        vocab = retunModelVocabulary(
            "Word2VecModel_Skipgram_Abstract_win_5_vec_300_20190731164233.bin")
        # print(vocab) ['#4363d8', '#911eb4', '#000075'][54686, 3920, 83382]
        print(len(vocab))
        a = (54686*360)/83382
        b = (3920*360)/83382
        # plotWordPerKB(["KG_GOSJG", "Ground truth"], [
        #               a, 360-a], ['#4363d8',  '#000075'])
        # plotWordPerKB(["KG_O", "Ground truth"], [
        #               b, 360-b], ['#4363d8',  '#000075'])
        plotAnalysis([54686], [3920], [83382], "KG_GOSJG", "KG_O", "Ground truth",
                     (" "),  "Knowledge graphs", "Number of words", " ")

    def dropRows(self):
        print("#######################")
        print(dropRowsWithEmptyProperty(
            "gramene_Oryza_sativa_japonica_genes_Propertiesdescription-entity_20190731191854.csv", "Outputs"))
        print("######################")


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
