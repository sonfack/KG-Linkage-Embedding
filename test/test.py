import unittest
import numpy as np 
from src.commons import findElementInListOfList, kMeans, calculateCenter

class TestLinkage(unittest.TestCase):
    
    def test_FindElementInListOfList(self):
        L = [[3,4,6],[1,8],[3,7,9]]
        e = 9
        self.assertTrue(findElementInListOfList(L,e))

    def test_kMeans(self):
        listOfVectors = [[1,1], [5,3], [2,1], [4,3], [5,4],[4,4]]
        k = 2
        kmeans = kMeans(listOfVectors, k)
        if k >= 6:
            self.assertEqual(kmeans, [])
        else:
            print(kmeans)
            self.assertTrue(kmeans[0] == [0,2] and kmeans[1] == [1,3,4,5], 'Bad classification')
        
    def test_calculateCenter(self):
        listOfVectors = [[1,1], [5,3], [2,1], [4,3], [5,4],[4,4]]
        listOfPoints = [0,3]
        self.assertTrue(np.all(calculateCenter(listOfPoints, listOfVectors) == [(1+4)/2,(1+3)/2]))



    
if __name__=="__main__":
    myTest = TestLinkage()
    #myTest.test_FindElementInListOfList()
    #myTest.test_kMeans()
    myTest.test_calculateCenter()
