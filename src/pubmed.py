"""
###
PubMed is a free search engine accessing primarily the MEDLINE database of references and abstracts on life sciences and biomedical topics. 
The United States National Library of Medicine (NLM) at the National Institutes of Health maintains the database as part of the Entrez system of information retrieval.
###
"""
"""
###
The Entrez Global Query Cross-Database Search System is a federated search engine, or web portal that allows users to search many discrete health sciences databases at the National Center for Biotechnology Information (NCBI) website.
###
"""
"""
Having difficulties to automaticaly install biopython, we downladed this verstion http://biopython.org/DIST/biopython-1.73.zip from their website 
and install it manualy. 
Install command : 
- unzip the package 
- enter in the folder 
- command -> pip3 install .
"""
"""
###
query_key
Query key. This integer specifies which of the UID lists attached to the given Web Environment will be used as input to EFetch. Query keys are obtained from the output of previous ESearch, EPost or ELInk calls. The query_key parameter must be used in conjunction with WebEnv.

WebEnv
Web Environment. This parameter specifies the Web Environment that contains the UID list to be provided as input to EFetch. Usually this WebEnv value is obtained from the output of a previous ESearch, EPost or ELink call. The WebEnv parameter must be used in conjunction with query_key.
###
"""
from Bio import Entrez
from Bio import Medline
import urllib.request as urllib2
#import urllib2
import sys

"""
email should be a valid email 
query should be a list of string
"""
def fetchByQuery(email,query,days=100):
    Entrez.email = email # you must give NCBI an email address
    if isinstance(query, list) and len(query) >= 2:
        query=" OR ".join(query)
    searchHandle=Entrez.esearch(db="pubmed", reldate=days, term=query, usehistory="y")
    searchResults=Entrez.read(searchHandle)
    searchHandle.close()
    webEnv=searchResults["WebEnv"]
    queryKey=searchResults["QueryKey"]
    print("Search results: ", searchResults)
    batchSize=10
    try:
        fetchHandle = Entrez.efetch(db="pubmed",
                                retmax=100,
                                rettype='full',
                                retmode="xml",
                                webenv=webEnv,
                                query_key=queryKey)
        xml_data=fetchHandle.read()
        print("#########################################")
        fetchHandle.close()
        if xml_data==None: 
            print(80*"*"+"\n")
            print("This search returned no hits")
        else:
            f=open("pmcXmlQuery.xml" ,"w")
            f.write(xml_data)
            f.close()
    except:
        return None


def fetchByPubmed(email, days=100):
    Entrez.email = email # you must give NCBI an email address
    handle = Entrez.efetch(db="pubmed",
                           id="19304878,19088134",
                           rettype='full',
                           retmode="xml")
    xml_data=handle.read()
    handle.close()
    if xml_data==None: 
        print(80*"*"+"\n")
        print("This search returned no hits")
    else:
        f=open("pmcXml.txt" ,"w")
        f.write(xml_data)
        f.close()

