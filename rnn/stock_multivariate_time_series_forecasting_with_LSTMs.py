# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:53:43 2020

@author: stephen.chen
"""

# from pandas import read_csv
# from pandas import DataFrame
# from pandas import concat
# from datetime import datetime
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential 
# from tensorflow.keras.layers import LSTM, Dense

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



# # load data
# def parse(x):
# 	return datetime.strptime(x, '%Y %m %d %H')


# dataset = read_csv('data\\raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
# dataset.drop('No', axis=1, inplace=True)
# # manually specify column names
# dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# dataset.index.name = 'date'
# # mark all NA values with 0
# dataset['pollution'].fillna(0, inplace=True)
# # drop the first 24 hours
# dataset = dataset[24:]
# # summarize first 5 rows
# print(dataset.head(5))
# # save to file
# dataset.to_csv('data\\pollution.csv')


from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv('data\\Tencent.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [ 0, 1, 2]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
 
# load dataset
dataset = read_csv('data\\Tencent.csv', header=0, index_col=0)
values = dataset.values


# integer encode direction
encoder = LabelEncoder()
values[:,2] = encoder.fit_transform(values[:,2])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[3,4]], axis=1, inplace=True)
print(reframed.head())



# split into train and test sets
values = reframed.values
n_train_hours = 1000
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

import tensorflow as tf
from tensorflow.keras import layers


# # design network
model = tf.keras.Sequential()
model.add(layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(layers.Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=60, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.figure(2)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# scaler.inverse_transform(yhat)
# scaler.inverse_transform(test_y)

print(scaler.data_max_)
print(scaler.data_min_)

b = 474.558
a = 47.436

yhat= yhat * (b - a) + a
test_y= test_y * (b - a) + a

pyplot.figure(3)
pyplot.subplot(3, 1, 1)
pyplot.plot(yhat)
pyplot.title('yhat', y=0.5, loc='right')
pyplot.subplot(3, 1, 2)
pyplot.plot(test_y)
pyplot.title('test_y', y=0.5, loc='right')
pyplot.subplot(3, 1, 3)
pyplot.plot(yhat-test_y)
pyplot.title('test_y', y=0.5, loc='right')
pyplot.legend()
pyplot.show()








# y_hat_train = model.predict(train_X)
# pyplot.figure()
# pyplot.subplot(2, 1, 1)
# pyplot.plot(y_hat_train)
# pyplot.title('y_hat_train', y=0.5, loc='right')
# pyplot.subplot(2, 1, 2)
# pyplot.plot(train_y)
# pyplot.title('train_y', y=0.5, loc='right')
# pyplot.legend()
# pyplot.show()

# print(yhat)