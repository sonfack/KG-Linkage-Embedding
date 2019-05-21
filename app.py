from src.commons import readDataFile, readUnformatedFile
from src.kgmanagement import getEntitiesPropertiesValue

if __name__== "__main__":
   # create extracted values from predicates of all entities in the file
   getEntitiesPropertiesValue("gramene_Oryza_sativa_japonica_genes.ttl")
   
   """
   dFille = readDataFile("reference.txt", folder="Texts")
   #dFille = readDataFile("gramene_Oryza_sativa_japonica_genes.csv", folder="Output")
   print(dFille.head())
   """
   
   
