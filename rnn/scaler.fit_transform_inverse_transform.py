# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:22:04 2020

@author: stephen.chen
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv

np.random.seed(7) 

val1= np.random.randint(100, size=(20,1))
val2= np.random.rand(20,1)
val3= np.random.rand(20,1)

a = 20
b = 30
val3 = val3*a + b



values = np.concatenate((val1, val2, val3), axis=1)
#values = np.c_[val1, val2, val3]



scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

recovered = scaler.inverse_transform(scaled)


filename = "data\\Tencent.csv"
# load dataset
dataset = read_csv(filename, header=0, index_col=0)
values = dataset.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
recovered = scaler.inverse_transform(scaled)