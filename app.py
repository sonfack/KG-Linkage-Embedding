from src.commons import readDataFile
from src.kgmanagement import getEntitiesPropertiesValue

if __name__== "__main__":
   getEntitiesPropertiesValue("gramene_Oryza_sativa_japonica_genes.ttl")
   exit()
   dFille = readDataFile("reference.table.txt")
   print(dFille)
