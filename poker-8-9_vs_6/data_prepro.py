# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 19:26:17 2017

@author: marry
"""
import os

os.chdir('C:\\Users\\marry\\OneDrive\\XLJ\\poker-8-9_vs_6')

#导入库
import pandas as pd
import numpy as np
#data=pd.read_table('kmeans_centroid.txt', header=None, delim_whitespace=True,index_col=0)
#data=data.T
#print (data)
#从去掉表头的txt文件读取数据
data=pd.read_table('poker_896.txt',names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','class'],sep=',')
#print(data)
#print(data['class']=='negative')

columns_index=list(data.columns)[:-1]

#样本不均衡的正负类分开，取数目多的类进行聚类
def distinct_data(data):
    positive_data=data.loc[data['class']=='positive']
    negative_data=data.loc[data['class']=='negative']
    #重置索引，从0开始
    positive_data=positive_data.reset_index(drop=True)
    negative_data=negative_data.reset_index(drop=True)
    if len(positive_data)>=len(negative_data):
        cluster_N=len(negative_data)
        cluster_data=positive_data.drop(['class'],axis=1)
        cluster_name='positive'
        other_data=negative_data
    else:
        cluster_N=len(positive_data)
        cluster_data=negative_data.drop(['class'],axis=1)
        cluster_name='negative'
        other_data=positive_data
    return cluster_N,cluster_data,cluster_name,other_data

#确定聚类个数，聚类数据，聚类名称，不聚类数据
cluster_N,cluster_data,cluster_name,other_data=distinct_data(data)
#print(positive_data)

from sklearn.cluster import KMeans
#调用Kmeans包进行聚类
estimator = KMeans(n_clusters=cluster_N)#构造聚类器
estimator.fit(cluster_data)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和

#print(centroids)
#1.均值聚类中心获取
centroids=pd.DataFrame(np.array(centroids),columns=columns_index)#聚类中心转换成dataframe
#negative_data['cluster']=label_pred#给数据打上聚类标签
#print(negative_data)

#计算两个向量的距离，用的是欧几里得距离
def distEclud(vecA, vecB):
     return np.sqrt(sum(np.power(vecA - vecB, 2)))

#2.以距离聚类中心最近的点代替聚类中心

#print(min_dist)
def reset_cluster_dist(cluster_data,cluster_N):
    min_dist=[100]*cluster_N
    minJ=np.zeros(cluster_N)
    for j in range(len(cluster_data)):
        #print(j)
        cluster=label_pred[j]
        #print(cluster)
        temp_dist=distEclud(np.array(cluster_data)[j],np.array(centroids)[cluster])
        #print(temp_dist)
        if(temp_dist<min_dist[cluster]):
            min_dist[cluster]=temp_dist
            minJ[cluster]=j
    minJ=[int(i) for i in minJ.tolist()]
    return minJ,min_dist

distJ,min_dist=reset_cluster_dist(cluster_data,cluster_N)
centroids_new_dist=pd.DataFrame(np.array(cluster_data)[distJ],columns=columns_index)

#使用信息增益加权欧式距离替换聚类中心
from im_gain import calc_ent_grap


gain_ent=dict(zip(columns_index,[0]*len(columns_index)))
#获取每个特征的信息增益，存储在字典中
gain_sum=0
for feature in columns_index:
    gain_ent[feature]=calc_ent_grap(data[feature],data['class'])
    gain_sum+=gain_ent[feature]

gain_weight=dict(zip(columns_index,[0]*len(columns_index)))
#将所有特征的信息增益归一化
gain_weight_sum=0
for feature in columns_index:
    gain_weight[feature]=gain_ent[feature]/gain_sum

def dist_gain(vecA, vecB,weight):
    temp_weight=np.array(list(weight.values()))
    return np.sqrt(sum(temp_weight*np.power(vecA - vecB, 2)))   
    
def reset_cluster_gain(cluster_data,cluster_N,weight):
    min_dist=[100]*cluster_N
    minJ=np.zeros(cluster_N)
    for j in range(len(cluster_data)):
        #print(j)
        cluster=label_pred[j]
        #print(cluster)
        temp_dist=dist_gain(np.array(cluster_data)[j],np.array(centroids)[cluster],weight)
        #print(temp_dist)
        if(temp_dist<min_dist[cluster]):
            min_dist[cluster]=temp_dist
            minJ[cluster]=j
    minJ=[int(i) for i in minJ.tolist()]
    return minJ,min_dist
 
gainJ,min_gain=reset_cluster_gain(cluster_data,cluster_N,gain_weight)
centroids_new_gain=pd.DataFrame(np.array(cluster_data)[gainJ],columns=columns_index)   
#print(centroids_new)
centroids_new_gain['class']=cluster_name
centroids['class']=cluster_name
centroids_new_dist['class']=cluster_name
#print(centroids)
#print(centroids_new)

#计算聚类样本重复率
rep_rate=list(np.array(distJ)-np.array(gainJ)).count(0)/cluster_N
print("离聚类中心最近点与加权最近点重复率")
print(rep_rate)

data_new1=pd.concat([other_data,centroids],ignore_index=True)
#print(data_new1)
data_new2=pd.concat([other_data,centroids_new_dist],ignore_index=True)
#print(data_new2)
data_new3=pd.concat([other_data,centroids_new_gain],ignore_index=True)
#print(data_new3)

data_new1.to_csv('poker_896_new1.arff',sep=',',header=False,index=False)
data_new2.to_csv('poker_896_new2.arff',sep=',',header=False,index=False)
data_new3.to_csv('poker_896_new3.arff',sep=',',header=False,index=False)

 

