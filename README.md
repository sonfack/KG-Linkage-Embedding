
# Application for Knowledge Graphs linkage using embeddings 


## How to run
### Create the corpus vector model (Skipgram for exple)
### Create for each knowledge base file (ttl) it properties file (CSV format).
For our case we have to call the src/kgmanagement/getEntitiesPropertiesValue funciton three times.
1. for our first knowledge base file
2. for our second knowledge base file
3. for our ground truth knowledge base file

def test_getEntitiesPropertiesValue(self):
     getEntitiesPropertiesValue("oryzabase_testold.ttl", None, "Datasets")

### Create for each knowledge base file (ttl) it frequency model
This frequency model will be use as weights for words vectors

### Create a file containing words and their tf or idf or tfidf

This is an intermediate stage to have for a give attribute of an entity a dictionary with words as keys an value of these keys (words) their vector representations from the embedding model of corpus.
model = "myModel.bin"
print(getAttributeVector(model, "gramene_Oryza_sativa_japonica_genes.csv",
                                  "OS02G0461200", entityProperty="description", folder="Texts"))

 create usable vectors of an entity
 model = "myModel.bin"
 fileNameTfIdf = "rowvocabularyTFIDFOf20190712114942.csv"
 vectorSize, attributeVector = getAttributeVector(model, "gramene_Oryza_sativa_japonica_genes.csv","OS02G0461200", entityProperty=["description", "label"], folder="Texts")

 compute similarity between two entities. The same model is used for both entities.
 For the frequency file, each entity uses the file of it knowledge based

     def test_similarity(self):
         model = "myModel.bin"
         fileNameTfIdf = "rowvocabularyTFIDFOf20190712114942.csv"
         vectorSize, attributeVector = getAttributeVector(model, "gramene_Oryza_sativa_japonica_genes.csv",
                                                          "OS02G0461200", entityProperty=["description", "label"], folder="Texts")
         vectorOne = usableAttributeVector(fileNameTfIdf, "tfidf",
                                           "OS02G0461200", attributeVector, vectorSize, folder="Outputs")
         print(computeSimilarity(vectorOne, vectorOne))
     compute complete similarity between two database files.
    def test_completeSimilarity(self):
         corpusModel = "myModel.bin"
         model = "tfidf"
         fileNameTfIdfOne = "rowvocabularyTFIDFOf20190712114942.csv"
         fileNameTfIdfTwo = "rowvocabularyTFIDFOf20190712114942.csv"
         databaseOne = "gramene_Oryza_sativa_japonica_genes.csv"
         databaseTwo = "gramene_Oryza_sativa_japonica_genes.csv"
         completeSimilarityOfDatasets(
             corpusModel, model, databaseOne, fileNameTfIdfOne, databaseTwo, fileNameTfIdfTwo)
   
    evaluation of the method
    `gFile = "oryzabase_ground_Propertiesdescription-entity-explanation-has_alternative_name-has_rap_identifier-has_synonym-has_tigr_identifier-has_trait_class-has_uniprot_accession-label-name_20190727202230.csv"
    gColumnName = ["entity", "has_rap_identifier"]

        rFile = "distancesCrossSimilaritydescription_tfidf_20190727174549.csv"
        rColumnName = ["orizabase_B", "orizabase_A"]
        threshold = [value*0.1 for value in range(10, 50, 5)]
        evaluation(gFile, gColumnName, rFile, rColumnName,threshold, 1, "Outputs", "Outputs", True)
`
        


## CHANGELOG 
