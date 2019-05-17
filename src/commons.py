import pandas as pd
import os
from chardet import detect


DATA_FOLDER = "data"
KB_FOLDER = os.path.join(DATA_FOLDER, "Datasets")
TEXT_FOLDER = os.path.join(DATA_FOLDER, "Texts")

def readDataFile(fileName):
    completeFileName = os.path.join(TEXT_FOLDER, fileName)
    dataFile = pd.read_csv(completeFileName, sep='\t')
    return dataFile

