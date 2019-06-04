from src.commons import readDataFile, readCompressDataFile, createtfIdfModel, wordsImportance, searchEntityInText, createCooccurenceMatrix, createListOfText
from src.kgmanagement import getEntitiesPropertiesValue
from src.embedding import trainingModel, cleaningDataset, createStopListFromFile, stoplist
from src.cooccurence import Cooccurrence


if __name__== "__main__":
   #create extracted values from predicates of all entities in the file
   #getEntitiesPropertiesValue("gramene_Oryza_sativa_japonica_genes.ttl")
   #trainingModel("gramene_Oryza_sativa_japonica_genes.csv")
   #createtfIdfModel("gramene_Oryza_sativa_japonica_genes.csv", "description", "Texts")
   #wordsImportance("gramene_Oryza_sativa_japonica_genes.csv", "description", "Texts")
   #readCompressDataFile("test.csv", "Texts")
   #searchEntityInText("gramene_Oryza_sativa_japonica_genes.csv", "label", "newdata.csv", "Abstract")
   #searchEntityInText("wordsLessImportance50pourcent.csv", "words", "newdata.csv", "Abstract")
  # stoplist = createStopListFromFile("wordsLessImportance50pourcent.csv", "words", "Texts")
   #cleanData = cleaningDataset(stoplist,"newdata.csv", "Abstract" )
   #trainingModel(stoplist, "newdata.csv", "Abstract")
   print("##################################################")
   listOfText = createListOfText("gramene_Oryza_sativa_japonica_genes_test.csv", "description" )
   createCooccurenceMatrix(stoplist, listOfText)
