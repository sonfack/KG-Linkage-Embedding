import pandas as pd
import os, logging, gzip
import pickle
from chardet import detect
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

DATA_FOLDER = "data"
KB_FOLDER = os.path.join(DATA_FOLDER, "Datasets")
TEXT_FOLDER = os.path.join(DATA_FOLDER, "Texts")
OUTPUT = os.path.join(DATA_FOLDER, "Outputs")
MODEL = os.path.join(DATA_FOLDER, "Models")

def readDataFile(fileName, folder="Texts"):
    if folder in "Texts" and len(folder) == len("Texts"):
        completeFileName = os.path.join(TEXT_FOLDER, fileName)
        dataFile = pd.read_csv(completeFileName, sep='\t', encoding="utf-8")
    elif folder in "Outputs" and len(folder) == len("Outputs"):
        completeFileName = os.path.join(OUTPUT, fileName)
        dataFile = pd.read_csv(completeFileName, sep='\t')
    elif folder in "Datasets" and len(folder) == len("Datasets"):
        completeFileName = os.path.join(KB_FOLDER, fileName)
        dataFile = pd.read_csv(completeFileName, sep='\t')
    elif folder in "Models" and len(folder) == len("Models"):
        completeFileName = os.path.join(MODEL, fileName)
        dataFile = pd.read_csv(completeFileName, sep='\t')
    return dataFile


def readCompressDataFile(fileName, folder="Texts"):
    if folder in "Texts" and len(folder) == len("Texts"):
        completeFileName = os.path.join(TEXT_FOLDER, fileName)
    elif folder in "Output" and len(folder) == len("Output"):
        completeFileName = os.path.join(OUTPUT, fileName)
    elif folder in "Datasets" and len(folder) == len("Datasets"):
        completeFileName = os.path.join(KB_FOLDER, fileName)
    with open(completeFileName, 'rb') as f:
        for i, line in enumerate(f):
            if (i % 100 == 0):
                logging.info("read {0} 100 of lines".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield simple_preprocess(line)


def createListOfText(fileName,columnName=None, folder="Texts"):
    dataFrame = readDataFile(fileName, folder)
    sentences = dataFrame[columnName]
    sentences.fillna("", inplace=True)
    sentences = sentences.values.tolist()
    return sentences

"""
This function creates a TF-IDF model 
"""
def createtfIdfModel(fileName, columnName, folder="Texts"):
    listOfText = createListOfText(fileName, columnName, folder)
    print(listOfText)
    tfIdfVectorizer = TfidfVectorizer()
    tfIdfVectorizer.fit(listOfText)
    with open(os.path.join(MODEL, "tfIdfVectorizer"), "+wb") as f:
        pickle.dump(tfIdfVectorizer, f, pickle.HIGHEST_PROTOCOL)
    f.close()


"""
This function gives TF-IDF of words in a text
"""
def wordsImportance(fileName, columnName, folder="Texts"):
    with open(os.path.join(MODEL, "tfIdfVectorizer"), "rb") as f:
        tfIdfVectorizer = pickle.load(f)
    f.close()
    listOfText = createListOfText(fileName, columnName, folder)
    
    X = tfIdfVectorizer.transform(listOfText)
    print(X)
    print(tfIdfVectorizer.get_feature_names())
    print(tfIdfVectorizer.vocabulary_)
    rows, cols = X.shape
    outputFile = os.path.join(OUTPUT, "wordsImportance.csv")
    importanceFile = open(outputFile, "a+")
    listOfAttributs = ["words", "descriptionID", "importance"]
    importanceFile.write("\t".join(listOfAttributs))
    importanceFile.write("\n")
    importanceFile.close()

    outputFile = os.path.join(OUTPUT, "wordsLessImportance.csv")
    lessImportanceFile = open(outputFile, "a+")
    listOfAttributs = ["words", "descriptionID", "importance"]
    lessImportanceFile.write("\t".join(listOfAttributs))
    lessImportanceFile.write("\n")
    lessImportanceFile.close()
    
    outputFile = os.path.join(OUTPUT, "wordsImportance.csv")
    importanceFile = open(outputFile, "a+")

    outputFile = os.path.join(OUTPUT, "wordsLessImportance.csv")
    lessImportanceFile = open(outputFile, "a+")

    for i in range(rows):
        for w,j in tfIdfVectorizer.vocabulary_.items():
            if X[i,j] >= 0.5:
                line = []
                line.append(w)
                line.append(str(i))
                line.append(str(X[i,j]))
                lineInfile = "\t".join(line)
                importanceFile.write(lineInfile)
                importanceFile.write("\n")
            elif X[i, j] != 0.0 and X[i, j] < 0.5:
                line = []
                line.append(w)
                line.append(str(i))
                line.append(str(X[i,j]))
                lineInfile = "\t".join(line)
                lessImportanceFile.write(lineInfile)
                lessImportanceFile.write("\n")
                
    importanceFile.close()
    lessImportanceFile.close()
