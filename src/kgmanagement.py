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
HAS_UNIPROT_ASSESSION = "has_uniprot_assession"

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
    outputList = {}
    # Output file of values of relevent properties of each entity
    outputFile = os.path.join(OUTPUT, kgFileName)
    graphFile = open(outputFile, "a+")
    graphFile.write("\t".join(listOfProperties))
    graphFile.close()
    listOfProperties.pop(0)
    completeKgFileName = os.path.join(KB_FOLDER, kgFileName)
    g = Graph()
    result = g.parse(completeKgFileName, format="n3")
    listOfSubjectsInGraph = g.subjects()
    for s in listOfSubjectsInGraph:
        listOfPropertiesInSubject = g.predicates(subject=s)
        for p in listOfPropertiesInSubject:
            graphProperty = p
            graphProperty = graphProperty.split("/")
            for givenProperty in listOfProperties:
                if  graphProperty[-1] in givenProperty:
                    listOfObjects = g.objects(subject=s, predicate=p)
                    res = [obj for obj in listOfObjects]
                    outputList[givenProperty]="$$".join(res)
                else:
                    outputList[givenProperty]=" "
        print(outputList)
        saveEntityAsFrameInFile(outputFile, listOfProperties, s, outputList)



def saveEntityAsFrameInFile(outputFile, listOfProperties, entityName, outputList):
    graphFile = open(outputFile, "a+")
    graphFile.write(str(entityName)+"\t"+"\t".join(str(objectValue) for objectValue in outputList.values()))
    graphFile.close()
