from src.commons import readCompressDataFile, createtfIdfModel, wordsImportance, searchEntityInText, createListOfText, generateTermDocumentMatrix, createCooccurrenceMatrix, createTfIdfAndBowModel, completeKmeans, stoplist, readDataFile
from src.kgmanagement import getEntitiesPropertiesValue
from src.embedding import trainingModel, cleaningDataset, createStopListFromFile


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
   listOfText = createListOfText("gramene_Oryza_sativa_japonica_genes.csv", "description" )
   print(listOfText)

   listOfVectors = createCooccurrenceMatrix(stoplist, listOfText)
   
   for k in range(2,50):
      print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
      print(completeKmeans([], listOfVectors, k, 100))
      print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
   
   #createTfIdfAndBowModel(listOfText)
   #generateTermDocumentMatrix(listOfText)
   
