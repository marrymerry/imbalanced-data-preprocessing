# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:28:30 2017

@author: marry
"""
#改变文件路径

import numpy as np




#1. 计算信息熵

def calc_ent(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

#2. 计算条件信息熵


def calc_condition_ent(x, y):
    """
        calculate ent H(y|x)
    """

    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        z=sub_y.reset_index(drop = True)
        temp_ent = calc_ent(z)
        ent += (float(z.shape[0]) / y.shape[0]) * temp_ent

    return ent

#3. 计算信息增益 ent_prap = H(Y) - H(Y|X)

def calc_ent_grap(x,y):
    """
        calculate ent grap
    """

    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap



#从去掉表头的txt文件读取数据
"""data=pd.read_table('yeast6.txt',names=['Mcg','Gvh','Alm','Mit','Erl','Pox','Vac','Nuc','class'],sep=', ')
columns=list(data.columns)[:-1]
gain_ent=dict(zip(columns,[0]*len(columns)))
#获取每个特征的信息增益，存储在字典中
gain_sum=0
for feature in columns:
    gain_ent[feature]=calc_ent_grap(data[feature],data['class'])
    gain_sum+=gain_ent[feature]

gain_weight=dict(zip(columns,[0]*len(columns)))
#将所有特征的信息增益归一化
gain_weight_sum=0
for feature in columns:
    gain_weight[feature]=gain_ent[feature]/gain_sum

def dist_gain(vecA, vecB,weight):
    temp_weight=np.array(list(weight.values()))
    return np.sqrt(sum(temp_weight*np.power(vecA - vecB, 2)))    
"""