from src.commons import readDataFile, readCompressDataFile, createtfIdfModel, wordsImportance, searchEntityInText
from src.kgmanagement import getEntitiesPropertiesValue
from src.embedding import trainingModel, cleaningDataset

if __name__== "__main__":
   #create extracted values from predicates of all entities in the file
   #getEntitiesPropertiesValue("gramene_Oryza_sativa_japonica_genes.ttl")
   #trainingModel("gramene_Oryza_sativa_japonica_genes.csv")
   #createtfIdfModel("gramene_Oryza_sativa_japonica_genes.csv", "description", "Texts")
   #wordsImportance("gramene_Oryza_sativa_japonica_genes.csv", "description", "Texts")
   #readCompressDataFile("test.csv", "Texts")
   #searchEntityInText("gramene_Oryza_sativa_japonica_genes.csv", "label", "newdata.csv", "Abstract")
   searchEntityInText("wordsLessImportance50pourcent.csv", "words", "newdata.csv", "Abstract")
