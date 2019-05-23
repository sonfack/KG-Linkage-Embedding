from src.commons import readDataFile,  readCompressDataFile
from src.kgmanagement import getEntitiesPropertiesValue
from src.embedding import trainingModel, cleaningDataset

if __name__== "__main__":
   # create extracted values from predicates of all entities in the file
   #getEntitiesPropertiesValue("gramene_Oryza_sativa_japonica_genes.ttl")
   #trainingModel("gramene_Oryza_sativa_japonica_genes.csv")
   createtfIdfModel("gramene_Oryza_sativa_japonica_genes.csv", description, "Texts")
