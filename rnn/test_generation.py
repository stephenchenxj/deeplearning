# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 11:14:06 2020

@author: stephen.chen
"""
# Importing dependencies numpy and keras
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils


"""
The text file is open, and all characters are converted to lowercase letters. 
In order to facilitate the following steps, we would be mapping each character 
to a respective number. This is done to make the computation part of the LSTM 
easier.
"""
# load text
filename = "data/macbeth.txt"
#filename = "data/short.txt"

text = (open(filename).read()).lower()

# mapping characters with integers
unique_chars = sorted(list(set(text)))

char_to_int = {}
int_to_char = {}

for i, c in enumerate (unique_chars):
    char_to_int.update({c: i})
    int_to_char.update({i: c})
    
    
"""
Data is prepared in a format such that if we want the LSTM to predict the ‘O’ 
in ‘HELLO’  we would feed in [‘H’, ‘E‘ , ‘L ‘ , ‘L‘ ] as the input and [‘O’] as 
the expected output. Similarly, here we fix the length of the sequence that we 
want (set to 50 in the example) and then save the encodings of the first 49 
characters in X and the expected output i.e. the 50th character in Y.
"""
# preparing input and output dataset
X = []
Y = []

for i in range(0, len(text) - 50, 1):
    sequence = text[i:i + 50]
    label =text[i + 50]
    X.append([char_to_int[char] for char in sequence])
    Y.append(char_to_int[label])
    
    
"""
A LSTM network expects the input to be in the form [samples, time steps, features] 
where samples is the number of data points we have, time steps is the number 
of time-dependent steps that are there in a single data point, features refers 
to the number of variables we have for the corresponding true value in Y. We 
then scale the values in X_modified between 0 to 1 and one hot encode our true 
values in Y_modified.
"""
# reshaping, normalizing and one hot encoding
X_modified = numpy.reshape(X, (len(X), 50, 1))
X_modified = X_modified / float(len(unique_chars))
Y_modified = np_utils.to_categorical(Y)


"""
A sequential model which is a linear stack of layers is used. The first layer 
is an LSTM layer with 300 memory units and it returns sequences. This is done 
to ensure that the next LSTM layer receives sequences and not just randomly 
scattered data. A dropout layer is applied after each LSTM layer to avoid 
overfitting of the model. Finally, we have the last layer as a fully connected 
layer with a ‘softmax’ activation and neurons equal to the number of unique 
characters, because we need to output one hot encoded result.
"""
# defining the LSTM model
model = Sequential()
model.add(LSTM(300, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')


# fitting the model
model.fit(X_modified, Y_modified, epochs=150, batch_size=30)

# picking a random seed
start_index = numpy.random.randint(0, len(X)-1)
print("start_index is " +str(start_index))
new_string = X[start_index]

# generating characters
for i in range(50):
    x = numpy.reshape(new_string, (1, len(new_string), 1))
    x = x / float(len(unique_chars))

    #predicting
    pred_index = numpy.argmax(model.predict(x, verbose=0))
    char_out = int_to_char[pred_index]
    seq_in = [int_to_char[value] for value in new_string]
    print(char_out)

    new_string.append(pred_index)
    new_string = new_string[1:len(new_string)]