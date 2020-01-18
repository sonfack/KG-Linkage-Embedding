import os
import matplotlib.pyplot as plt
from datetime import datetime
from src.commons import readDataFile, createListOfTextFromListOfFileNameByRow
from src.predefined import OUTPUT


def evaluation(groundFile, groundColumnName, resultFile, resultColumnName, threshold=None, distance=None, groundFileFolder="Outputs", resultFileFolder="Outputs", plot=False):
    """
    This funciton takes the groud file and the result file, and returns for a percentage of correct match entities from the 
    ground file. It does this for each distances used in the result file.
    Parameters:
    :param groundFile: is the ground truth file name containing the matches of entities from both knowledge based files 
    :param groundColumnName: the column names (02) corresponding to the matches.
    :param resultFile: is the result file from cross calculations of distances 
    :param resultColumnName: is the column of interes from the result file 
    :param threshold: the value that the distances should satisfied.
    default -> None 
    :param distance: is the type of distance been used.
    default -> None 
    1 -> euclidean 
    2 -> cosine 




    :param plot: states if the threshold-precision graph should be ploted
    True -> plot graph 
    False -> do not plot graph 
    """
    groundFrame = readDataFile(groundFile, groundFileFolder)
    groundRows, groundCols = groundFrame.shape
    resultFrame = readDataFile(resultFile, resultFileFolder)
    resultRows, resultCols = resultFrame.shape
    extractedGround = groundFrame[groundColumnName]
    distanceInfo = resultFile.split("_")
    countMatch = 0
    outputevaluationFile = "evaluation"+str(
        datetime.now()).replace(":", "").replace("-", "").replace(" ", "").split(".")[0]+".txt"
    f = open(os.path.join(OUTPUT, outputevaluationFile), "a+")
    f.write("Ground file \n")
    f.write(groundFile)
    f.write("\n")
    f.write("Result file \n")
    f.write(resultFile)
    f.write("\n")
    f.write("Corpus Model \n")
    f.wrtite(distanceInfo[2])
    f.write("Corpus Model window size \n")
    f.write(distanceInfo[4])
    f.write("Corpus Model vector dimension \n")
    f.write(distanceInfo[6])
    f.write("Corpus Model attribute \n")
    f.write(" ".join(distanceInfo[8].split("-")))
    f.write("Weight coef \n")
    f.write(distanceInfo[10])
    f.write("\n")
    f.close()
    if isinstance(threshold, int) or isinstance(threshold, float):
        for index, row in extractedGround.iterrows():
            couple = [row[groundColumnName[0]], row[groundColumnName[1]]]
            print("### groud couple")
            print(couple)
            print("###")
            matchFrame = resultFrame[resultFrame[resultColumnName[0]] == couple[0]]
            matchValues = matchFrame.values
            if not matchFrame.empty and matchValues[0][1] and matchValues[0][1] == couple[1] and matchValues[0][distance+1] >= threshold:
                countMatch += 1
                print("### countMatch")
                print(countMatch)
                print("###")
                print("### matchFrame")
                print(matchFrame.values)
                print("###")
        f = open(os.path.join(OUTPUT, outputevaluationFile), "a+")
        f.write("Recall: \n")
        recall = countMatch/groundRows
        f.write(str(recall))
        f.write("\n")
        f.write("Precision: \n")
        precision = countMatch/resultRows
        f.write(str(precision))
        f.close()
        return precision, recall
    elif isinstance(threshold, list) and plot == True:
        listOfPrecision = []
        print("### list of threshold")
        print(threshold)
        print("###")
        for th in threshold:
            print("### th in threshold")
            print(th)
            print("###")
            print()
            prec, rec = evaluation(groundFile, groundColumnName, resultFile,
                                   resultColumnName, th, distance, groundFileFolder, resultFileFolder, False)
            listOfPrecision.append(prec)

        print("### listOfPrecision")
        print(listOfPrecision)
        print("###")
        fig = plt.figure()
        plt.plot(threshold, listOfPrecision, 'ro')
        plt.axis([0, max(threshold), 0, 1])
        # plt.show()
        fig.savefig(os.path.join(OUTPUT, "evaluation"+"_plot_"+str(
            datetime.now()).replace(":", "").replace("-", "").replace(" ", "").split(".")[0]+".png"))


def analysisValues(csvKB, csvKBFolder):
    dataFrame = readDataFile(csvKB, csvKBFolder)
    print("### number of entities ")
    numberOfEntities, cols = dataFrame.shape
    print(numberOfEntities)
    print("###")
    print("### missing values ")
    numberOfMissingValue = (dataFrame == '').sum(axis=1).sum(axis=0)
    print(numberOfMissingValue)
    print("###")
    return numberOfEntities, numberOfMissingValue


def numberOfImpEntity(wordImpCSV, wordImpCSVFolder):
    dataFrame = readDataFile(wordImpCSV, wordImpCSVFolder)
    df = dataFrame['entity'].nunique()
    print("### entity groups")
    print(df)
    print("###")
    return int(df)


def returnNumberOfVectorPerKB():
    """
    This function retuns the number of vectors used for given dataset
    """
    pass


def returnDatasetVocabulary(csvKB, columnName, csvKBFolder):
    """
    This function returns the vocabulary of a dataset
    """
    listOfEntity, listOfSentences = createListOfTextFromListOfFileNameByRow(
        csvKB, columnName, None, "Outputs")
    return listOfSentences
