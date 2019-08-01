import os
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
HAS_RAP_IDENTIFIER = "has_rap_identifier"

LISTOFPROPERTIES = [DESCRIPTION, HAS_RAP_IDENTIFIER, HAS_TIGR_IDENTIFIER, LABEL, HAS_UNIPROT_ASSESSION,
                    NAME, EXPLANATION, HAS_SYNONYM, HAS_ALTERNATIVE_NAME, HAS_TRAIT_CLASS]


DATA_FOLDER = "data"
KB_FOLDER = os.path.join(DATA_FOLDER, "Datasets")
TEXT_FOLDER = os.path.join(DATA_FOLDER, "Texts")
GROUND_FOLDER = os.path.join(DATA_FOLDER, "Grounds")
OUTPUT = os.path.join(DATA_FOLDER, "Outputs")
MODEL = os.path.join(DATA_FOLDER, "Models")


TFIDFMODEL = "tfIdfVectorizer20190621143653"
#TFMODEL = "KBrowtfIdfVectorizer20190706171104"
TFMODEL = "tfVectorizer20190621143651"
