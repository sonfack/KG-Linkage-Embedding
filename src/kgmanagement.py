import os 
from rdflib import Graph, URIRef, Literal
from src.commons import DATA_FOLDER, KB_FOLDER, OUTPUT, checkIfEntityInDataset


"""
1. Put you KG files in the data folder
Bellow are the properties we are looking on each entity
"""
DESCRIPTION = "description"
HAS_TIGR_IDENTIFIER = "has_tigr_identifier"
LABEL = "label"
HAS_UNIPROT_ASSESSION = "has_uniprot_accession"
NAME = "name"
EXPLANATION = "explanation"
HAS_SYNONYM = "has_synonym"
HAS_ALTERNATIVE_NAME = "has_alternative_name"
HAS_TRAIT_CLASS = "has_trait_class"

LISTOFPROPERTIES = [DESCRIPTION, HAS_TIGR_IDENTIFIER, LABEL, HAS_UNIPROT_ASSESSION, NAME, EXPLANATION, HAS_SYNONYM, HAS_ALTERNATIVE_NAME, HAS_TRAIT_CLASS]


"""
2. This function outputs a frame containing values of selected attributs of a given graph and save in a csv file
"""
def getEntitiesPropertiesValue(kgFileName, properties=None):
    if properties is None:
        # We ordered our default properties
        listOfProperties = LISTOFPROPERTIES
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
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print(outputList)
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        saveEntityAsFrameInFile(outputFile, s, outputList,listOfProperties)


        
"""
3. This function saves values of properties in the file given as parameter
"""        
def saveEntityAsFrameInFile(outputFile, entity, outputList, listOfProperties):
    entityUri = entity.split("/")[-1]
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(entityUri)
    print(outputFile)
    print(checkIfEntityInDataset(entityUri, "entity", outputFile))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    
    if not checkIfEntityInDataset(entityUri, "entity", outputFile): 
        outputListSorted = []
        graphFile = open(outputFile, "a+")
        outputList["entity"] = entityUri 
        print("##################################################################")
        if len(outputList) != len(listOfProperties):
            for property in listOfProperties:
                if property not in outputList.keys():
                    outputList[property]=" "
        print(outputList)
        print("##################################################################")
        for key in sorted(outputList.keys()):
            propertyValue = outputList[key].split()
            listPropertyValue = [theValue.split("/")[-1] for theValue in propertyValue]
            for index  in range(len(listPropertyValue)):
                if "LOC" in listPropertyValue[index]:
                    theValue = listPropertyValue[index].split("_")[-1]
                    listPropertyValue[index] = theValue
            outputListSorted.append(" ".join(listPropertyValue))
        linesInFile = "\t".join(outputListSorted)
        graphFile.write(linesInFile)
        graphFile.write("\n")
        graphFile.close()
