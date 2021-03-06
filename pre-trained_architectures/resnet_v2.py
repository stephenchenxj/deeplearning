# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:58:37 2020

@author: stephen.chen
"""
   
#import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_explain.core.activations import ExtractActivations
from tensorflow.keras.applications.resnet_v2 import decode_predictions
# %matplotlib inline    
    
#load pre trained model
model=tf.keras.applications.resnet_v2.ResNet152V2(weights='imagenet',include_top=True)
#Summary of Model
#print(model.summary)
print(model.summary())


#loading and preprocessing image
IMAGE_PATH='../test/lucky.jpg'
img=tf.keras.preprocessing.image.load_img(IMAGE_PATH,target_size=(224,224))
img=tf.keras.preprocessing.image.img_to_array(img)
#view the image
plt.imshow(img/255.)


import requests
#fetching labels from Imagenet  
response=requests.get('https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
imgnet_map=response.json()
imgnet_map={v[1]:k for k, v in imgnet_map.items()}
#make model predictions
img=tf.keras.applications.resnet_v2.preprocess_input(img)
predictions=model.predict(np.array([img]))
#decode_predictions(predictions,top=5)
print(decode_predictions(predictions,top=5))