import os 
from rdflib import Graph, URIRef, Literal
from src.commons import DATA_FOLDER, KB_FOLDER

"""
Put you KG files in the data folder
"""

def getEntitiesPropertiesValue(kgFileName):
    completeKgFileName = os.path.join(KB_FOLDER, kgFileName)
    g = Graph()
    result = g.parse(completeKgFileName, format="n3")
    for s, p, o in g:
        print(p)
