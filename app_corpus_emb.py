import os
import traceback
import unittest
from datetime import datetime
from multiprocessing import Process
import numpy as np
#from src.pubmed import fetchByPubmed, fetchByQuery
#from src.baseline import getListOfText, createDistance
from src.commons import wordsImportance, cleaningText, stoplist, createFrequencyModel, createListOfTextFromListOfFileNameByRow, partitionDataset, readDataFile
from src.embedding import trainingModel, plotPCA, getAttributeVector, usableAttributeVector, computeSimilarity, completeSimilarityOfDatasets, getWordAggregationFromFile, cleaningDataset
from src.evalaluate import evaluation
from src.kgmanagement import getEntitiesPropertiesValue
from src.predefined import DATA_FOLDER, LISTOFPROPERTIES, TEXT_FOLDER


# 0 corpus embedding
def corpusEmbedding():
    listOfModel = []
    listOfModelFolder = []
    dataSetFile = os.path.join(TEXT_FOLDER,"newdata.csv")
    t = stoplist
    #vectorSize = [25, 50, 100, 150, 200, 250, 300]
    vectorSize = [25, 50, 100]
    #windowSize = [2, 3, 5]
    windowSize = [3, 5]
    for vec in vectorSize:
        for win in windowSize:
            folder, model = trainingModel(
                t, dataSetFile, "Abstract", 1, vec, win, 1, 3, None, "row", "Texts")
            #listOfModel.append(model)
            #listOfModelFolder.append(folder)
    #return listOfModelFolder, listOfModel

# 1 Create for each knowledge base file (ttl) it properties file.
# For our case we have to call the src/kgmanagement/getEntitiesPropertiesValue funciton three times
# - 1 for our first knowledge base file
# - 2 for our second knowledge base file
# - 3 for our ground truth knowledge base file

def running_corpusEmbedding(log):
    try:
        corpusEmbedding()
    except Exception:
        traceback.print_exc(file=log)



if __name__ == "__main__":

    log = open("Log_eval"+str(
        datetime.now()).replace(":", "").replace("-", "").replace(" ", "").split(".")[0] + ".txt", "a+")

    running_corpusEmbedding(log)
