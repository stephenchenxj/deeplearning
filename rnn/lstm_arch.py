# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 21:59:15 2020

@author: stephen.chen
"""

from datetime import datetime
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

from keras.layers.core import Dropout

from pandas import read_csv
from matplotlib import pyplot
 

import tensorflow as tf
from tensorflow.keras import layers

from keras.layers import LSTM
# # design network
model = Sequential()
#model.add(LSTM(2, return_sequences=True, input_shape=(1, 1)))
model.add(LSTM(1, input_shape=(1, 1)))
# model.add(layers.LSTM(20, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(20, return_sequences=True, recurrent_dropout = 0.2))
# model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(20, return_sequences=True, recurrent_dropout = 0.2))
# model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(20, return_sequences=True, recurrent_dropout = 0.2))
# model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(20, return_sequences=True, recurrent_dropout = 0.2))
# model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(20, return_sequences=True, recurrent_dropout = 0.2))
# model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(20, return_sequences=True, recurrent_dropout = 0.2))
# model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(20, return_sequences=True, recurrent_dropout = 0.2))
# model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(20)) 
# model.add(layers.Dropout(0.2))
model.add(Dense(1))

model.summary()