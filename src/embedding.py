"""
   This module is about functions related to embeddings
"""
import os
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
from gensim.summarization import summarize, textcleaner
from src.commons import readDataFile, readCompressDataFile, stoplist, createVocabulary, createListOfTextFromListOfFileNameByColumn, createListOfTextFromListOfFileNameByRow
from collections import Counter
from src.predefined import LISTOFPROPERTIES, MODEL, OUTPUT

cores = multiprocessing.cpu_count()  # Count the number of cores in a computer


def createStopListFromFile(fileName, columnName=None, by="column", folder="Texts"):
    """
       This function is used to create a list of stopwords from a given column or columns from a data file ( .csv)  
    """
    df = readDataFile(fileName, folder)
    words = df[columnName].values.tolist()
    return set(words)


"""
This funciton removes define stop words given in the list stoplist from documents from a databasefile  ( dataset is a csv file)
Documents can be taken by row/column
if by row, each row is consider as a document
if by column, each column is consider as a document
"""


def cleaningDataset(stopList, dataSetFile, columnName, by="row", folder="Texts"):
    if by == "row" or by is None:
        listOfSentences = createListOfTextFromListOfFileNameByRow(
            dataSetFile, columnName)
    elif by == "column":
        listOfSentences = createListOfTextFromListOfFileNameByColumn(
            dataSetFile, columnName)
    print("###")
    print(listOfSentences)
    print("###")
    cleanData = []
    for sentence in range(len(listOfSentences)):
        # Removing stopwords and punctuations
        sentence = [word for word in " ".join(listOfSentences[sentence]).split()
                    if word not in stopList]
        cleanData.append(sentence)
    return cleanData


def getAttributeVector(modelFile, dataBaseFile, entity, model="tfidf", entityProperty=None, folder="Outputs"):
    """
       This function gets a database name, a given entity, a given property/list of properties of the entity.
       It returns the vecteur representing the entity from the embedding  
    """
    df = readDataFile(dataBaseFile, folder)
    listOfColumns = list(df.columns)
    print(listOfColumns)
    rows, cols = df.shape
    modelFile = os.path.join(MODEL, modelFile)
    cFile = open(modelFile, "rb")
    model = pickle.load(myModel)

    modelVocabulary = list(model.wv.vocab.keys())
    print(modelVocabulary)
    print("Shape of data frame: ", df.shape)
    dataBaseRow = 0
    meetEntity = False
    while dataBaseRow in range(rows) and meetEntity == False:
        listRow = df.iloc[dataBaseRow, :]
        if str(listRow[1]) == str(entity):
            meetEntity = True
        else:
            dataBaseRow += 1
    listRow = df.iloc[dataBaseRow, :]
    # if entity in dataBaseFile
    if dataBaseRow in range(rows) and isinstance(entityProperty, str) and entityProperty in listOfColumns:
        try:
            colIndex = listOfColumns.index(entityProperty)
            print("Property index", colIndex)
            attribute, attributeSize = createVocabulary(
                stoplist, listRow[colIndex])
            attributeVector = {}
            attributeVocabulary = {}
            for attr in attribute:
                if attr in modelVocabulary:
                    print("# ", attr)
                    attributeVocabulary[attr] = model[attr]
            attributeVector[entityProperty] = attributeVocabulary
            print("Attribute:", attributeVector)
            return attributeVector
        except:
            print("PROPERTY : ", entityProperty, "NOT IN DATABASE")
    elif dataBaseRow in range(rows) and isinstance(entityProperty, list):
        print("list")
        listOfAttributesVectors = []
        try:
            for propertyInList in entityProperty:
                if propertyInList in listOfColumns:
                    print(propertyInList)
                    colIndex = listOfColumns.index(propertyInList)
                    print("Property index ", colIndex)
                    attributeVector = {}
                    attribute, attributeSize = createVocabulary(
                        stoplist, listRow[colIndex])
                    attributeVocabulary = {}
                    for attr in attribute:
                        print("# ", attr)
                        if attr in modelVocabulary:
                            attributeVocabulary[attr] = model[attr]
                    attributeVector[propertyInList] = attributeVocabulary
                listOfAttributesVectors.append(attributeVector)
            print(listOfAttributesVectors)
            return listOfAttributesVectors
        except:
            print("PROPERTY : ", entityProperty, "NOT IN DATABASE")


"""
This function returns a vector given the dictionary of an attribute with dictionary vectors of the key words that constitute them. 
If is mean at the  coefficientVector/aggregate the resultant vector is the sum of all the words vectors 
Else we use a pondarate sum of the vectors and their coefficients ( tf, idf) 
"""


def usableAttributeVector(myModel, entityUri, attributeVector, fileName,  aggregate="mean"):
    myModelSource = os.path.join(MODEL, myModel)
    model = Word2Vec.load(myModelSource)
    if aggregate == "mean":
        if isinstance(attributeVector, dict):
            sumVector = np.zeros(model.vector_size, dtype="float64")
            for attribute in attributeVector:
                for keyword in attributeVector[attribute]:
                    v = np.array(
                        attributeVector[attribute][keyword], dtype="float64")
                    sumVector += v
            return np.divide(sumVector, len(attributeVector))
        elif isinstance(attributeVector, list):
            sumVector = np.zeros(model.vector_size, dtype="float64")
            for eachAttribut in attributeVector:
                sumVector += usableAttributeVector(myModel,
                                                   eachAttribut, aggregate)
            return sumVector
    elif aggregate == "tf":
        if isinstance(attributeVector, dict):
            sumVector = np.zeros(model.vector_size, dtype="float64")
            for attribute in attributeVector:
                for keyword in attributeVector[attribute]:
                    v = np.array(
                        attributeVector[attribute][keyword], dtype="float64")
                    sumVector += np.multiply(
                        v, getWordAggregationFromFile(fileName, keyword, folder))
            return sumVector
        elif isinstance(attributeVector, list):
            pass
    elif aggregate == "idf" or aggregate == "tf-idf":
        if isinstance(attributeVector, dict):
            sumVector = np.zeros(model.vector_size, dtype="float64")
            for attribute in attributeVector:
                for keyword in attributeVector[attribute]:
                    v = np.array(
                        attributeVector[attribute][keyword], dtype="float64")
                    sumVector += np.multiply(v, getWordAggregationFromFile(
                        fileName, keyword, entityIndex, folder))
            return sumVector
        elif isinstance(attributeVector, list):
            sumVectorFinal = np.zeros(model.vector_size, dtype="float64")
            for attribute in attributeVector:
                sumVectorFinal += usableAttributeVector(
                    myModel, entityUri, attribute, fileName, aggregate)
            return sumVectorFinal


"""
This function gets a word tf or tf-idf from file, this is done by setting the entity index value ( case of tf-idf) or none ( case of tf)
NB: word in parameter should be a word from vocabulary, if word not in file the funciton returns 0 as it coefficient 
This function uses the database file to map entityIndex and entity row by row 
"""


def getWordAggregationFromFile(fileName, word, entityIndex=None, folder="Texts"):
    fileFrame = readDataFile(fileName, folder)
    if entityIndex is None:
        listOfWords = fileFrame["words"].values.tolist()
        if word in listOfWords:
            ind = listOfWords.index(word)
            print(word, ind)
            return float(fileFrame.loc[ind, "tf"])
        else:
            return 0
    elif entityIndex is not None:
        rows, cols = fileFrame.shape
        row = 0
        while row in range(rows):
            rowInfo = fileFrame.loc[row,
                                    fileFrame.columns.tolist()].values.tolist()
            if rowInfo[0] == word and (int(rowInfo[1]) == entityIndex):
                print(rowInfo)
                return float(rowInfo[-1])
            else:
                row += 1
        return 0


"""
This function takes two usable vectors of entities and computes
their cosine similarity
"""


def computeSimilarity(entityVectorOne, entityVectorTwo):
    v1 = np.array([entityVectorOne])
    v2 = np.array([entityVectorTwo])
    cosine_similarity1 = np.dot(entityVectorOne, entityVectorTwo)
    cosine_similarity2 = linalg.norm(
        entityVectorOne)*linalg.norm(entityVectorTwo)
    return linalg.norm(np.subtract(v1, v2)), cosine_similarity1 / cosine_similarity2


"""
This function takes two datasets(csv format) and returns a file containing cross similarity of all their entities
"""


def completeSimilarityOfDatasets(myModel, dataBaseFileOne, dataBaseFileTwo, properties=None, folder="Outputs"):
    dfOne = readDataFile(dataBaseFileOne, folder)
    dfTwo = readDataFile(dataBaseFileTwo, folder)
    rowsOne, colsOne = dfOne.shape
    rowsTwo, colsTwo = dfTwo.shape

    listOfAttributs = properties

    outputCombineFile = os.path.join(OUTPUT, "distances"+str(datetime.now()).replace(
        ":", "").replace("-", "").replace(" ", "").split(".")[0]+".csv")

    fileOne = dataBaseFileOne.split(".csv")

    fileTwo = dataBaseFileTwo.split(".csv")

    characteristicCombineFile = open(outputCombineFile, "a+")
    characteristicCombineFile.write(
        "\t".join([fileOne[0], fileTwo[0], "euclidean", "cosine"]))
    characteristicCombineFile.write("\n")
    characteristicCombineFile.close()
    if properties is None:
        listOfAttributs = LISTOFPROPERTIES
    for indexOne in range(rowsOne):
        for indexTwo in range(rowsTwo):
            listRowOne = dfOne.iloc[indexOne, :]
            listRowTwo = dfTwo.iloc[indexTwo, :]

            attributeVectorOne = getAttributeVector(
                myModel, dataBaseFileOne, str(listRowOne[1]), listOfAttributs)
            attributeVectorTwo = getAttributeVector(
                myModel, dataBaseFileTwo, str(listRowTwo[1]), listOfAttributs)
            entityVectorOne = usableAttributeVector(
                myModel, attributeVectorOne)
            entityVectorTwo = usableAttributeVector(
                myModel, attributeVectorTwo)
            euclideanDistance, cosineDistance = computeSimilarity(
                entityVectorOne, entityVectorTwo)
            print(str(listRowOne[1]), " - ", str(listRowTwo[1]),
                  " == ", euclideanDistance, cosineDistance)

            characteristicCombineFile = open(outputCombineFile, "a+")
            characteristicCombineFile.write("\t".join([str(listRowOne[1]), str(
                listRowTwo[1]), str(euclideanDistance), str(cosineDistance)]))
            characteristicCombineFile.write("\n")
            characteristicCombineFile.close()


"""
This function looks for the nearest neighbours of an entity from on knowledge base in another knowledge base.
parameters: 
- second knowledge base 
- the entity/entities (uri) for which we are looking for nearest neighbours.
- threshold
It returns a file containing all the found neighbours 
"""


def getNearestEntitiesOfEntity(firstKBVectorFile, secondKBVectorFile, entityFromFirstKB, threshold=0, frequencyModel="tfidf", folder="Texts"):
    firstFileFrame = readDataFile(firstKBVectorFile, folder)
    secondFileFrame = readDataFile(secondKBVectorFile, folder)
    listOfEntityInFirstFileFrame = firstFileFrame.loc[firstFileFrame["entity"]
                                                      == entityFromFirstKB]
    if len(listOfEntityInFirstFileFrame) == 1:
        firstEntityVector = listOfEntityInFirstFileFrame[0][1]
        print("Vector : ", firstEntityVector)
        listOfSecondFileVector = secondFileFrame[["entity", "vector"]]
        for row in listOfSecondFileVector:
            computeSimilarity(firstEntityVector, row[1])
    else:
        print("Many occurrences")


"""
# Training the model

size: (default 100) The number of dimensions of the embedding, e.g. the length of the dense vector to represent each token (word).

window: (default 5) The maximum distance between a target word and words around the target word.

min_count: (default 5) The minimum count of words to consider when training the model; words with an occurrence less than this count will be ignored.

workers: (default 3) The number of threads to use while training.

sg: (default 0 or CBOW) The training algorithm, either CBOW (0) or skip gram (1).

total_examples: (int) Count of sentences;

epochs: (int) - Number of iterations (epochs) over the corpus - [10, 20, 30]

progress_per: (int, optional) – Indicates how many words to process before showing/updating the progress.
"""


def trainingModel(stopwords, dataSetFile, columnName, minCountOfAWord=1, embeddingDimension=100, windowSize=5, architecture=1, numberOfTreads=3):
    print("File : ", dataSetFile)
    if dataSetFile and columnName:
        dataSet = cleaningDataset(stopwords, dataSetFile, columnName)
    else:
        print("No dataset or column")
        """
        dataSet = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'], ['this', 'is', 'the', 'second', 'sentence'], [
            'yet', 'another', 'sentence'], ['one', 'more', 'sentence'], ['and', 'the', 'final', 'sentence']]
        """
    """
    doc: (str) – Input document.
    deacc: (bool, optional) – Remove accent marks from tokens using deaccent()?
    min_len: (int, optional) – Minimum length of token (inclusive). Shorter tokens are discarded.
    max_len: (int, optional) – Maximum length of token in result (inclusive). Longer tokens are discarded.
    #gensim.utils.simple_preprocess(doc, deacc=False, min_len=2, max_len=15)
    """
    model = Word2Vec(dataSet, min_count=minCountOfAWord, size=embeddingDimension,
                     window=windowSize, sg=architecture, workers=cores)
    model.train(dataSet, total_examples=model.corpus_count,
                epochs=30, report_delay=1)
    model.save(os.path.join(MODEL, "Word2VecModel"+str(datetime.now()).replace(":",
                                                                               "").replace("-", "").replace(" ", "").split(".")[0] + ".bin"))
    print("End training")


"""
Plote the PCA on a given number of components ( numberOfComponent)
"""


def plotPCA(myModel, numberOfComponent=2):
    model = Word2Vec.load(myModel)
    vect_putative = model["putative"]
    print(vect_putative[0])
    print(list(model.wv.vocab.keys()))
    """
    X = model[model.wv.vocab]
    pca = PCA(numberOfComponent)
    result = pca.fit_transform(X)

    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()
    """
