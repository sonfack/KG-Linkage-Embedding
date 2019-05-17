import os 
from rdflib import Graph, URIRef, Literal
from src.commons import DATA_FOLDER, KB_FOLDER

"""
Put you KG files in the data folder
"""
DESCRIPTION = "description"
HAS_TIGR_IDENTIFIER = "has_tigr_identifier"
LABEL = "label"
HAS_UNIPROT_ASSESSION = "has_uniprot_assession"


def getEntitiesPropertiesValue(kgFileName):
    completeKgFileName = os.path.join(KB_FOLDER, kgFileName)
    g = Graph()
    result = g.parse(completeKgFileName, format="n3")
    for s, p, o in g:
        graphProperty = s
        graphProperties = graphProperties.split("/")
        if LABEL in graphProperties[-1]:
            pass 
