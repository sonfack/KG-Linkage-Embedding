import os
from datetime import datetime
from rdflib import Graph, URIRef, Literal
from src.predefined import LISTOFPROPERTIES, OUTPUT, DATA_FOLDER
from src.commons import readDataFile

"""
2. This function outputs a csv vile containing values of selected attributs of a given graph
NB: The entity property is added automaticaly to the list of properties
"""


def getEntitiesPropertiesValue(kgFileName, properties=None, kgFileNameFolder="Datasets", outputFileFolder="Outputs"):
    listOfProperties = []
    if properties is None:
        # We ordered our default properties
        listOfProperties = LISTOFPROPERTIES
    if isinstance(properties, str):
        listOfProperties.append(properties)
    elif isinstance(properties, list):
        listOfProperties = properties
    listOfProperties.insert(0, "entity")
    listOfProperties.sort()
    print("### listOfProperties")
    print(listOfProperties)
    print("###")

    # Output file of values of relevent properties of each entity
    kgFileName = kgFileName.split(".")
    outputFile = kgFileName[0]+"_Properties"+"-".join(listOfProperties)+"_"+str(datetime.now()).replace(
        ":", "").replace("-", "").replace(" ", "").split(".")[0]+".csv"
    outputFileDirectory = os.path.join(DATA_FOLDER, outputFileFolder)
    graphFile = open(os.path.join(outputFileDirectory, outputFile), "a+")
    print("### graphFile")
    print(os.path.join(outputFileDirectory, outputFile))
    print("###")
    graphFile.write("\t".join(listOfProperties))
    graphFile.write("\n")
    graphFile.close()
    directory = os.path.join(DATA_FOLDER, kgFileNameFolder)
    completeKgFileName = os.path.join(directory, ".".join(kgFileName))
    print("###")
    print("File name: ", completeKgFileName)
    print("###")
    g = Graph()
    result = g.parse(completeKgFileName, format="n3")
    listOfSubjectsInGraph = g.subjects()
    for s in listOfSubjectsInGraph:
        outputList = {}
        print("###")
        print("subject", s)
        print("###")
        listOfPropertiesInSubject = g.predicates(subject=s)
        for p in listOfPropertiesInSubject:
            graphProperty = p
            print("###")
            print(graphProperty)
            print("###")
            graphProperty = graphProperty.split("/")
            nameOfProperty = graphProperty[-1]
            if nameOfProperty in listOfProperties or nameOfProperty.split("#")[-1] in listOfProperties:
                listOfObjects = g.objects(subject=s, predicate=p)
                res = [obj for obj in listOfObjects]
                if len(res) > 0:
                    if nameOfProperty.find("#label") != -1:
                        outputList["label"] = " ".join(res)
                    elif "description" in nameOfProperty and len("description") == len(nameOfProperty):
                        description = []
                        for desc in res:
                            for d in desc.split(";"):
                                if d.find("cDNA", 0, 4) != -1 or (d.find("Os", 0, 2) != 1 and d.find("protein")) or d.find("TrEMBL") or d.find("Acc:", 0, 4):
                                    if d.find(",") != -1:
                                        for sd in d.split(","):
                                            description.append(sd)
                                    else:
                                        description.append(d)
                        outputList[nameOfProperty] = " ".join(description)
                    else:
                        outputList[nameOfProperty] = " ".join(res)
                else:
                    outputList[nameOfProperty] = " "
                print("### outputList")
                print(outputList)
                print("###")
            else:
                print("### nameOfProperty")
                print(nameOfProperty, " is not in this entity properties")
                print("###")
        saveEntityAsFrameInFile(
            outputFile, s, outputList, listOfProperties, outputFileFolder)
    return outputFileFolder, outputFile


"""
3. This function saves values of properties in the file given as parameter
"""


def saveEntityAsFrameInFile(outputFile, entity, outputList, listOfProperties, outputFileFolder):
    entityUri = entity.split("/")[-1]
    print("### entity")
    print(entityUri)
    print(outputFile)
    print(checkIfEntityInDataset(entityUri, "entity", outputFile, outputFileFolder))
    print("###")
    if not checkIfEntityInDataset(entityUri, "entity", outputFile, outputFileFolder):
        outputListSorted = []
        outputFileDirectory = os.path.join(DATA_FOLDER, outputFileFolder)
        graphFile = open(os.path.join(outputFileDirectory, outputFile), "a+")
        outputList["entity"] = entityUri
        print("### outputList")
        if len(outputList) != len(listOfProperties):
            for property in listOfProperties:
                if property not in outputList.keys():
                    outputList[property] = " "
        print(outputList)
        print("###")
        for key in sorted(outputList.keys()):
            propertyValue = outputList[key].split()
            listPropertyValue = [theValue.split(
                "/")[-1] for theValue in propertyValue]
            for index in range(len(listPropertyValue)):
                if "LOC" in listPropertyValue[index]:
                    theValue = listPropertyValue[index].split("_")[-1]
                    listPropertyValue[index] = theValue
            outputListSorted.append(" ".join(listPropertyValue))
        linesInFile = "\t".join(outputListSorted)
        graphFile.write(linesInFile)
        graphFile.write("\n")
        graphFile.close()
        return outputFile


"""
This function verifies if a given entity URI is present in data base file(csv). 
It returns true is the entity is present and false otherwise 
"""


def checkIfEntityInDataset(entityURI, entityAttributeName, datasetFile, datasetFileFolder="Outputs"):
    df = readDataFile(datasetFile, datasetFileFolder)
    listAttrib = list(df[entityAttributeName])
    listAttribute = [str(item) for item in listAttrib]
    if len(listAttribute) >= 1:
        if str(entityURI) in listAttribute:
            return True
        # elif int(entityURI) in listAttribute:
        #     return True
        else:
            return False
    else:
        return False
