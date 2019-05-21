import os 
from rdflib import Graph, URIRef, Literal
from src.commons import DATA_FOLDER, KB_FOLDER, OUTPUT

"""
1. Put you KG files in the data folder
Bellow are the properties we are looking on each entity
"""
DESCRIPTION = "description"
HAS_TIGR_IDENTIFIER = "has_tigr_identifier"
LABEL = "label"
HAS_UNIPROT_ASSESSION = "has_uniprot_accession"

"""
2. This function outputs a frame containing values of selected attributs of a given graph
"""

def getEntitiesPropertiesValue(kgFileName, properties=None):
    if properties is None:
        # We ordered our default properties
        listOfProperties = [DESCRIPTION, HAS_TIGR_IDENTIFIER, LABEL, HAS_UNIPROT_ASSESSION]
    else:
        listOfProperties = properties
    listOfProperties.insert(0, "entity" )
    listOfProperties.sort()
    print("#############################################################################################################################")
    print(listOfProperties)
    print("#############################################################################################################################")
    # Output file of values of relevent properties of each entity
    kgFileName = kgFileName.split(".")
    outputFile = os.path.join(OUTPUT, kgFileName[0]+".csv")
    graphFile = open(outputFile, "a+")
    graphFile.write("\t".join(listOfProperties))
    graphFile.write("\n")
    graphFile.close()
    completeKgFileName = os.path.join(KB_FOLDER, ".".join(kgFileName))
    g = Graph()
    result = g.parse(completeKgFileName, format="n3")
    listOfSubjectsInGraph = g.subjects()
    for s in listOfSubjectsInGraph:
        outputList = {}
        print("subject",s)
        listOfPropertiesInSubject = g.predicates(subject=s)
        for p in listOfPropertiesInSubject:
            graphProperty = p
            print(graphProperty)
            graphProperty = graphProperty.split("/")
            nameOfProperty = graphProperty[-1]
            if  nameOfProperty in listOfProperties or nameOfProperty.find("#label") != -1:
                listOfObjects = g.objects(subject=s, predicate=p)
                res = [obj for obj in listOfObjects]
                if len(res) > 0:
                    if nameOfProperty.find("#label") != -1:
                        outputList["label"] = " ".join(res)
                    else:
                        outputList[nameOfProperty] = " ".join(res)
                else:
                    outputList[nameOfProperty] = " "
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print(outputList)
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        saveEntityAsFrameInFile(outputFile, s, outputList,listOfProperties)

        
"""
This function saves values of properties in the file given as parameter
"""        
def saveEntityAsFrameInFile(outputFile, entity, outputList, listOfProperties):
    outputListSorted = []
    graphFile = open(outputFile, "a+")
    outputList["entity"] = entity.split("/")[-1]
    print("#############################################################################################################################")
    if len(outputList) != len(listOfProperties):
        for property in listOfProperties:
            if property not in outputList.keys():
                outputList[property]=" "
    print(outputList)
    print("#############################################################################################################################")
    for key in sorted(outputList.keys()):
        propertyValue = outputList[key].split()
        if len(propertyValue)>= 2:
            listPropertyValue = [theValue.split("/")[-1] for theValue in propertyValue]
            outputListSorted.append(" ".join(listPropertyValue))
        else:
            outputListSorted.append(" ".join(propertyValue))
    linesInFile = "\t".join(outputListSorted)
    graphFile.write(linesInFile)
    graphFile.write("\n")
    graphFile.close()
