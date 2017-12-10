# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:26:42 2017

@author: gaomengrui
"""

#coding=utf-8
 2 from numpy import *
 3 import pandas as pd 
 
 4 def loadDataSet(fileName):
 5     dataMat = []
 6     fr = open(fileName)
 7     for line in fr.readlines():
 8         curLine = line.strip().split('\t')
 9         fltLine = map(float, curLine)
10         dataMat.append(fltLine)
11     return dataMat
12     
13 #计算两个向量的距离，用的是欧几里得距离
14 def distEclud(vecA, vecB):
15     return sqrt(sum(power(vecA - vecB, 2)))
16 
17 #随机生成初始的质心（ng的课说的初始方式是随机选K个点）    
18 def randCent(dataSet, k):
19     n = shape(dataSet)[1]
20     centroids = mat(zeros((k,n)))
21     for j in range(n):
22         minJ = min(dataSet[:,j])
23         rangeJ = float(max(array(dataSet)[:,j]) - minJ)
24         centroids[:,j] = minJ + rangeJ * random.rand(k,1)
25     return centroids
26     
27 def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
28     m = shape(dataSet)[0]
29     clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
30                                       #to a centroid, also holds SE of each point
31     centroids = createCent(dataSet, k)
32     clusterChanged = True
33     while clusterChanged:
34         clusterChanged = False
35         for i in range(m):#for each data point assign it to the closest centroid
36             minDist = inf
37             minIndex = -1
38             for j in range(k):
39                 distJI = distMeas(centroids[j,:],dataSet[i,:])
40                 if distJI < minDist:
41                     minDist = distJI; minIndex = j
42             if clusterAssment[i,0] != minIndex: 
43                 clusterChanged = True
44             clusterAssment[i,:] = minIndex,minDist**2
45         print centroids
46         for cent in range(k):#recalculate centroids
47             ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
48             centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
49     return centroids, clusterAssment
50     
51 def show(dataSet, k, centroids, clusterAssment):
52     from matplotlib import pyplot as plt  
53     numSamples, dim = dataSet.shape  
54     mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
55     for i in xrange(numSamples):  
56         markIndex = int(clusterAssment[i, 0])  
57         plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
58     mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
59     for i in range(k):  
60         plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
61     plt.show()
62       
63 def main():
64     dataMat = mat(loadDataSet('testSet.txt'))
65     myCentroids, clustAssing= kMeans(dataMat,4)
66     print myCentroids
67     show(dataMat, 4, myCentroids, clustAssing)  
68     
69     
70 if __name__ == '__main__':
71     main()