
# Knowledge Graphs linkage using embeddings 

## How the method runs
### Create the corpus vector model (Skipgram for exple)
The corpus used here is related to the tow knoledge bases and is supposed to have words form the properties of entities of both knowledge bases.

### Create for each knowledge base file (ttl) it properties file (CSV format)
The properties file of a knowledge base file is an extraction of entities of knowledge base file originaly on ttl format to a csv format.
Note that entities are identified with their URI.
For our case, we have to call the src/kgmanagement/getEntitiesPropertiesValue funciton three times.
1. for our first knowledge base file
2. for our second knowledge base file
3. for our ground truth knowledge base file
eg: getEntitiesPropertiesValue("oryzabase_testold.ttl", None, "Datasets")

### Create for each knowledge base file it frequency model
This frequency model will be use as weights for words vectors.
We can compute 03 types of frequence models:
1. tf: term frequency
2. idf: inverse document frequency
3. tfidf: term frquency inverser document frequency

### Create a file containing words and their tf or idf or tfidf

This is an intermediate stage to have for a give attribute of an entity a dictionary with words as keys an value of these keys (words) their vector representations from the embedding model of corpus.

```
model = "myModel.bin"
print(getAttributeVector(model, "gramene_Oryza_sativa_japonica_genes.csv",
                                  "OS02G0461200", entityProperty="description", folder="Texts"))
```
#### create usable vectors of an entity
We add or sum up word of each property of an entity of a knowledge base file, by multiplying each word found in the corpus model vector representation by the word frequency (tf, idf,tf-idf) 

```
model = "myModel.bin"
fileNameTfIdf = "rowvocabularyTFIDFOf20190712114942.csv"
vectorSize, attributeVector = getAttributeVector(model, "gramene_Oryza_sativa_japonica_genes.csv","OS02G0461200", entityProperty=["description", "label"], folder="Texts")
```
Compute similarity between two entities. The same model is used for both entities.
For the frequency file, each entity uses the file of it knowledge based. 
```
model = "myModel.bin"
fileNameTfIdf = "rowvocabularyTFIDFOf20190712114942.csv"
vectorSize, attributeVector = getAttributeVector(model, "gramene_Oryza_sativa_japonica_genes.csv",
                                                          "OS02G0461200", entityProperty=["description", "label"], folder="Texts")
         vectorOne = usableAttributeVector(fileNameTfIdf, "tfidf",
                                           "OS02G0461200", attributeVector, vectorSize, folder="Outputs")
         print(computeSimilarity(vectorOne, vectorOne))
```
Compute complete similarity between two database files.
```
corpusModel = "myModel.bin"
model = "tfidf"
fileNameTfIdfOne = "rowvocabularyTFIDFOf20190712114942.csv"
fileNameTfIdfTwo = "rowvocabularyTFIDFOf20190712114942.csv"
databaseOne = "gramene_Oryza_sativa_japonica_genes.csv"
databaseTwo = "gramene_Oryza_sativa_japonica_genes.csv"
completeSimilarityOfDatasets(corpusModel, model, databaseOne, fileNameTfIdfOne, databaseTwo, fileNameTfIdfTwo)
```
#### Evaluation of the method
Find out the precision en recall of the method. This is based on a threshold used as minimum value that the distance that should exist between two entities from the knowledge bases.
```
    gFile = "oryzabase_ground_Propertiesdescription-entity-explanation-has_alternative_name-has_rap_identifier-has_synonym-has_tigr_identifier-has_trait_class-has_uniprot_accession-label-name_20190727202230.csv"
    gColumnName = ["entity", "has_rap_identifier"]

        rFile = "distancesCrossSimilaritydescription_tfidf_20190727174549.csv"
        rColumnName = ["orizabase_B", "orizabase_A"]
        threshold = [value*0.1 for value in range(10, 50, 5)]
        evaluation(gFile, gColumnName, rFile, rColumnName,threshold, 1, "Outputs", "Outputs", True)
```
        


## CHANGELOG 
