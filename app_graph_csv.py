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
from src.predefined import DATA_FOLDER, LISTOFPROPERTIES


def kgToCVSFile():
    kgFiles = ["RAP-MSU_2019-08-29.ttl", "all.locus_brief_info.7.ttl", "agrold.oryza_sativa.ttl", "agrold.oryza_sativa_xrefs.ttl" ]
    for kgfile in kgFiles:
        getEntitiesPropertiesValue(kgfile)


def test_evaluation(log):
    try:
        getEntitiesPropertiesValue()
        
    except Exception:
        traceback.print_exc(file=log)


if __name__ == "__main__":

    log = open("Log_eval"+str(
        datetime.now()).replace(":", "").replace("-", "").replace(" ", "").split(".")[0] + ".txt", "a+")

    test_evaluation(log)
