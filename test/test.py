import unittest
import numpy as np 
from src.commons import findElementInListOfList, kMeans, calculateCenter, compareSelectedVectors, completeKmeans


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
            self.assertTrue(kmeans[0] == [0,2] and kmeans[1] == [1,3,4,5], 'Bad classification')
        
    def test_calculateCenter(self):
        listOfVectors = [[1,1], [5,3], [2,1], [4,3], [5,4],[4,4]]
        listOfPoints = [1,3,4,5]
        self.assertTrue(np.all(calculateCenter(listOfPoints, listOfVectors) == [(5+4+5+4)/4,(3+3+4+4)/4]))

    def test_compareSelectedVectors(self):
        v1 = [[0,2], [1,3,4]]
        v2 = [[0,2],[1,3],[4]]
        self.assertFalse(compareSelectedVectors(v1,v2))
        v3 = [[0,2], [1,3,4]]
        v4 = [[0,2],[1,4,3]]
        self.assertTrue(compareSelectedVectors(v3,v4))


    def test_completeKmeans(self):
        listOfVectors = [[1,2], [2,1], [2,4], [4,2], [4,3], [5,3], [5,2], [2,3], [1,3], [1,1], [3,1], [3,3]]
#=[[1,1],[2,2],[2,3],[3,1],[4,1],[4,2],[5,2],[5,3],[5,5],[4,5],[6,2],[6,3],[6,4]]
        #[[1,1],[5,3], [2,1], [4,3], [5,4],[4,4]]
        k = 4
        #v4 = [[0,2], [1,3,4]]
        l, c =completeKmeans([], listOfVectors, k, 9)
        #self.assertTrue(compareSelectedVectors(l, v4))

        
if __name__=="__main__":
    myTest = TestLinkage()
    #myTest.test_FindElementInListOfList()
    #myTest.test_kMeans()
    #myTest.test_calculateCenter()
    #myTest.test_compareSelectedVectors()
    myTest.test_completeKmeans()
