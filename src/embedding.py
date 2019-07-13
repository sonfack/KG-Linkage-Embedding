"""
   This module is about functions related to embeddings
"""
import os
import pickle
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


def cleaningDataset(stopList, dataSetFile, columnName, by="row", folder="Texts"):
    """
       This funciton removes define stop words given in the list stoplist from documents from a databasefile  ( dataset is a csv file)
       Documents can be taken by row/column
       if by row, each row is consider as a document
       if by column, each column is consider as a document
    """
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


def getAttributeVector(myModel, dataBaseFile, entity, entityProperty=None, folder="Outputs"):
    """
       This function get a
       :param myModel: the embedding model from the corpus 
       :param dataBaseFile: The knowledge based file name (csv fromat) 
       :param entity: the URI of the entity we are interested on 
       :param entityProperty: a given property/list of properties of the entity.
       :param folder: is the name of the folder present in the data folder and containing the knowledge based file use by this function.
       It returns the vecteur representing the entity from the embedding  
"""
    df = readDataFile(dataBaseFile, folder)
    listOfColumns = list(df.columns)
    rows, cols = df.shape
    myModel = os.path.join(MODEL, myModel)
    model = Word2Vec.load(myModel)
    vectorSize = model.vector_size
    modelVocabulary = list(model.wv.vocab.keys())
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
            attributeVector = {}
            attribute, attributeSize = createVocabulary(
                stoplist, listRow[colIndex])
            attributeVocabulary = {}
            for attr in attribute:
                if attr in modelVocabulary:
                    print("# ", attr)
                    attributeVocabulary[attr] = model[attr]
            attributeVector[entityProperty] = attributeVocabulary
            print("Attribute:", attributeVector)
            return vectorSize, attributeVector
        except:
            print("PROPERTY : ", entityProperty, "NOT IN DATABASE")
    elif dataBaseRow in range(rows) and isinstance(entityProperty, list):
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
            return vectorSize, listOfAttributesVectors
        except:
            print("PROPERTY : ", entityProperty, "NOT IN DATABASE")


def usableAttributeVector(frequencyModelFile, model, entity, attributeVector, vectorSize, folder="Outputs"):
    print(np.ones(3)*4+np.ones(3))
    """
       :param frequencyModelFile: is the csv file containing words and their frequencies (tf/idf/tfidf)
       :param model : is the model being used (tf/idf/tfidf)
       :param entity : is the URI of the entity we are look for it vector 
       :param attributeVector : a list of  dictionary returned from getAttributeVector and containing relevent words from attribute of an entity 
       :param vectorSize: is the size of word vector from the embedding model

       This function returns a vector given the dictionary of an attribute with dictionary vectors of the key words that constitute them.  
    """
    print("### attributeVector")
    print(attributeVector)
    print("###")
    frequencyDataFrame = readDataFile(frequencyModelFile, folder)
    if model == "idf" or model == "IDF":
        modelValue = vocabCount.idf_
        modelVocabulary = countMatrix.get_feature_names()

    elif model in ["TF-IDF", "tf-idf", "TFIDF", "tfidf", "TF", "tf"]:
        entityDataFrame = frequencyDataFrame.loc[frequencyDataFrame.loc[:,
                                                                        "entity"] == entity, :]
        print("### entity frame")
        print(entityDataFrame)
        print("###")
        listOfWords = entityDataFrame.loc[:, "word"].values
        print("### list of words")
        print(listOfWords)
        print("###")
        if isinstance(attributeVector, dict):
            sumVector = np.zeros(vectorSize, dtype="float64")
            for word in listOfWords:
                for attribute in attributeVector:
                    print("###")
                    print("attribute", attribute)
                    v = np.zeros(vectorSize, dtype="float64")
                    if word in attributeVector[attribute]:
                        print("word", word)
                        v = np.array(
                            attributeVector[attribute][word], dtype="float64")
                        coef = entityDataFrame.loc[entityDataFrame.loc[:,
                                                                       "word"] == word].values
                        print("coefficient value",
                              coef[0, 2])
                        v = v*coef[0, 2]
                        print("### vectore multiply by coef")
                        print(v)
                        print("###")
                    sumVector += v
                    print("###")
            return sumVector
        elif isinstance(attributeVector, list):
            finalVector = np.zeros(vectorSize, dtype="float64")
            for attribute in attributeVector:
                finalVector += usableAttributeVector(
                    frequencyModelFile, model, entity, attribute, vectorSize, folder)
            return finalVector


def getWordAggregationFromFile(fileName, word, entityIndex=None, folder="Texts"):
    """
       This function gets a word tf or tf-idf from file, this is done by setting the entity index value ( case of tf-idf) or none ( case of tf)
       NB: word in parameter should be a word from vocabulary, if word not in file the funciton returns 0 as it coefficient 
       This function uses the database file to map entityIndex and entity row by row 
    """
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


def trainingModel(stopwords, dataSetFile, columnName, minCountOfAWord=1, embeddingDimension=100, windowSize=5, architecture=1, numberOfTreads=3):
    """
       Training a new Word2Vec model 

       size: (default 100) The number of dimensions of the embedding, e.g. the length of the dense vector to represent each token (word).

       window: (default 5) The maximum distance between a target word and words around the target word.

       min_count: (default 5) The minimum count of words to consider when training the model; words with an occurrence less than this count will be ignored.

       workers: (default 3) The number of threads to use while training.

       sg: (default 0 or CBOW) The training algorithm, either CBOW (0) or skip gram (1).

       total_examples: (int) Count of sentences;

       epochs: (int) - Number of iterations (epochs) over the corpus - [10, 20, 30]

       progress_per: (int, optional) – Indicates how many words to process before showing/updating the progress.

       dataSetFile: the knowledge base file that will be use (csv file)

       columnName: the column from the knowledge based file that will be used for embedding.

       columnName can be a given column or a list of list of column
       Exple of dataset 
       dataSet = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'], ['this', 'is', 'the', 'second', 'sentence'], [
        'yet', 'another', 'sentence'], ['one', 'more', 'sentence'], ['and', 'the', 'final', 'sentence']]
       doc: (str) – Input document.
       deacc: (bool, optional) – Remove accent marks from tokens using deaccent()?
       min_len: (int, optional) – Minimum length of token (inclusive). Shorter tokens are discarded.
       max_len: (int, optional) – Maximum length of token in result (inclusive). Longer tokens are discarded.
       gensim.utils.simple_preprocess(doc, deacc=False, min_len=2, max_len=15)
    """
    print("###")
    print("File : ", dataSetFile)
    print("###")
    if dataSetFile and columnName:
        dataSet = cleaningDataset(stopwords, dataSetFile, columnName)
    else:
        print("###")
        print("No dataset or column")
        print("###")
    model = Word2Vec(dataSet, min_count=minCountOfAWord, size=embeddingDimension,
                     window=windowSize, sg=architecture, workers=cores)
    model.train(dataSet, total_examples=model.corpus_count,
                epochs=30, report_delay=1)
    model.save(os.path.join(MODEL, "Word2VecModel"+str(datetime.now()).replace(":",
                                                                               "").replace("-", "").replace(" ", "").split(".")[0] + ".bin"))
    print("End training")


def plotPCA(myModel, numberOfComponent=2):
    """
       Plote the PCA on a given number of components ( numberOfComponent)
    """
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
