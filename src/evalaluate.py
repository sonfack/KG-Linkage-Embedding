import os
import matplotlib.pyplot as plt
from datetime import datetime
from src.commons import readDataFile
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
    """
    groundFrame = readDataFile(groundFile, groundFileFolder)
    groundRows, groundCols = groundFrame.shape
    resultFrame = readDataFile(resultFile, resultFileFolder)
    resultRows, resultCols = resultFrame.shape
    extractedGround = groundFrame[groundColumnName]
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
    f.close()
    if isinstance(threshold, int):
        for index, row in extractedGround.iterrows():
            couple = [row[groundColumnName[0]], row[groundColumnName[1]]]
            print("### groud couple")
            print(couple)
            print("###")
            matchFrame = resultFrame[resultFrame[resultColumnName[0]] == couple[0]]
            print("### matchFrame")
            print(matchFrame.values)
            print("###")
            matchValues = matchFrame.values
            if not matchFrame.empty and matchValues[0][1] and matchValues[0][1] == couple[1] and matchValues[0][distance+1] >= threshold:
                countMatch += 1
                print("### countMatch")
                print(countMatch)
                print("###")
        f = open(os.path.join(OUTPUT, outputevaluationFile), "a+")
        f.write("Recall: \n")
        f.write(str(countMatch/groundRows))
        f.write("\n")
        f.write("Precision: \n")
        f.write(str(countMatch/resultRows))
        f.close()
        return countMatch/groundRows
    elif isinstance(threshold, list):
        listOfPrecision = []
        for alpha in threshold:
            precision = evaluation(groundFile, groundColumnName, resultFile,
                                   resultColumnName, alpha, "Outputs", "Outputs", plot=False)
            listOfPrecision.append(precision)
        fig = plt.figure()
        plt.plot(threshold, precision, 'ro')
        plt.axis([0, max(threshold), 0, 1])
        # plt.show()
        fig.savefig("evaluation"+"_".join(listOfAttributs)+model+str(
            datetime.now()).replace(":", "").replace("-", "").replace(" ", "").split(".")[0]+".png")
