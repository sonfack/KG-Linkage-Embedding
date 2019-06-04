import pandas as pd
import os, logging, gzip
import pickle
import numpy as np
from  numpy import linalg
from collections import Counter
from chardet import detect
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

DATA_FOLDER = "data"
KB_FOLDER = os.path.join(DATA_FOLDER, "Datasets")
TEXT_FOLDER = os.path.join(DATA_FOLDER, "Texts")
OUTPUT = os.path.join(DATA_FOLDER, "Outputs")
MODEL = os.path.join(DATA_FOLDER, "Models")

def readDataFile(fileName, folder="Texts"):
    dataFile = ""
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
    outputFile = os.path.join(OUTPUT, "newdata.csv")
    with open(completeFileName, 'r') as f:
        for i, line in enumerate(f):
            for sl in line.split("\\n"):
                for tsl in sl.split("\\t"):
                    newdata = open(outputFile, "a+")
                    newdata.write(tsl) 
                    newdata.write("\t")
                    newdata.close()
                newdata = open(outputFile, "a+")
                newdata.write("\n")
                newdata.close() 

                
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


"""   
#Read CSV File
def read_csv(fileName, json_file, format):
    completeFileName = os.path.join(TEXT_FOLDER, fileName)
    outputFile = os.path.join(OUTPUT, json_file)
    csv_rows = []
    with open(completeFileName) as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames
        for row in reader:
            csv_rows.extend([{title[i]:row[title[i]] for i in range(len(title))}])
        write_json(csv_rows, outputFile, format)

#Convert csv data into json and write it
def write_json(data, json_file, format):
    with open(json_file, "w") as f:
        if format == "pretty":
            f.write(json.dumps(data, sort_keys=False, indent=4, separators=(',', ': '),encoding="utf-8",ensure_ascii=False))
        else:
            f.write(json.dumps(data))
"""

def searchEntityInText(kgFile, kgAttrib, textFile, textAttrib, folder="Texts"):
    dfKg = readDataFile(kgFile)
    dfText = readDataFile(textFile)
    listOfKg = dfKg[kgAttrib].tolist()
    listOfText = dfText[textAttrib].tolist()
    count = 0
    setOfWords = []
    for text in listOfText:
        for word in listOfKg:
            if str(text).find(str(word)) != -1:
                setOfWords.append(str(word))
                count += 1
    print(Counter(setOfWords))
    print("Vocabulary of Kg", len(setOfWords))
    print(count)


def cleaningText(stoplist, Text):
    #Removing stopwords and punctuations
    sentence = [word for word in Text.split() if word not in stoplist]
    return sentence


def createVocabulary(stoplist, Text):
    numberOfVocab = 0
    vocab = []
    if isinstance(Text, list):
        for text in Text:
            textWord = cleaningText(stoplist, text)
            vocab += textWord
    elif isinstance(Text, str):
        textWord = cleaningText(stoplist, Text)
        vocab = textWord
    vocab = list(set(vocab))
    numberOfVocab = len(vocab)
    return vocab, numberOfVocab


def createCooccurenceMatrix(stoplist, listOfText, windSize=1):
    vocab, numberOfVocab = createVocabulary(stoplist, listOfText)
    print(vocab)
    print(numberOfVocab)
    cooccurenceMat = []
    for x in range(len(vocab)):
        y = x+1
        while y in range(len(vocab)):
            print(vocab[x],' - ', vocab[y])
            for text in listOfText:
                text = cleaningText(stoplist, text)
                if vocab[x] in text and vocab[y] in text:
                    i = text.count(vocab[x])
                    index =  text.index(vocab[x])
                    if i == 1 and index+windSize < len(text) and text[index+windSize] == vocab[y]:
                        row = [0]*len(vocab)
                        row[y] += 1
                        print(row)
                    elif i > 1:
                        row = [0]*len(vocab)
                        for k  in range(len(text)):
                            if text[k] == vocab[x] and k+windSize < len(text) and text[k+windSize] == vocab[y] :
                                row[y] += 1
                        print(row)
            y +=1
            cooccurenceMat.append(row)
        
    return cooccurenceMat

def findElementInListOfList(listOfList, element):
    find = False
    index = 0
    while find == False and index  in range(len(listOfList)):
        if element in listOfList[index]:
            find = True
        else:
            index += 1
    if index not in range(len(listOfList)):
        return False
    else:
        return True

def calculateCenter(listOfPoints, cooccurenceMat):
    centerPoint = np.array([0]*len(cooccurenceMat[0]))
    for i in listOfPoints:
        centerPoint = centerPoint + np.array(cooccurenceMat[i])
    centerPoint =  centerPoint/len(listOfPoints)
    print('Calculate center',centerPoint)
    return centerPoint


"""
compareSelectedVectors returns true if the two vectors are different
"""
def compareSelectedVectors(firstSelectedVector, secondSelectedVector):
    if len(firstSelectedVector) == len(secondSelectedVector):
        i = 0
        difference = False 
        while i in range(len(firstSelectedVector)) and difference == False:
            if len(firstSelectedVector[i]) == len(secondSelectedVector[i]):
                for j in range(len(firstSelectedVector[i])):
                    if firstSelectedVector[i][j] not in secondSelectedVector[i]:
                        difference = True
                i += 1
            else:
                difference = True
        if i not in range(len(firstSelectedVector)):
            return True
        else:
            return False
    else:
        return False 


def kMeans(cooccurrenceMat, k, listOfCenters=[]):
    print('Cooccurrence Matrix', cooccurrenceMat)
    rows = len(cooccurrenceMat)
    # list of centers also represent the classes
    selectedVectors = []
    selectedVect = {}
    distancesFromCenters = {}
    if k <= rows:
        if not listOfCenters:
            # Initilization centre points 
            for i in range(k):
                listOfCenters.append(cooccurrenceMat[i])
        # Classification
        for j in range(rows):
            for i in range(len(listOfCenters)):
                distancesFromCenters[i] = linalg.norm(np.subtract(listOfCenters[i],cooccurrenceMat[j]))
            ind = min(distancesFromCenters, key=lambda k: distancesFromCenters[k])
            selectedVect[j] = ind
        # computing centers
        listOfCenters = list()
        for j in range(k):
            v = []
            for key , val in selectedVect.items():
                if int( selectedVect[key]) == j:
                    v.append(key)
            listOfCenters.insert(j,list(calculateCenter(v, cooccurrenceMat)))
            selectedVectors.insert(j, v)
        print('New centers', listOfCenters)
    print('Selected vectors end', selectedVectors)
    return selectedVectors, listOfCenters


def completeKmeans(selectedVectors, cooccurrenceMat, k, itteration, listOfCenters=[]):
    print('Itteration', itteration)
    newSelectedVectors,newListOfCenters = kMeans(cooccurrenceMat, k, listOfCenters)
    if compareSelectedVectors(newSelectedVectors, selectedVectors) == False and itteration-1 > 0:
        completeKmeans(newSelectedVectors, cooccurrenceMat, k, itteration-1, newListOfCenters)
    else:
        return newSelectedVectors
