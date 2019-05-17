from rdflib import Graph, URIRef, Literal

def getEntitiesPropertiesValue(kgFilename):
        g = Graph()
	result = g.parse(kgFilename)
	for s, p, o in g:
		 if (s, p, o) not in g:
			raise Exception("It better be!")
