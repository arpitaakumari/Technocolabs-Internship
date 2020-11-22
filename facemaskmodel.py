# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 19:05:36 2020

@author: Arpita Kumari
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf

#get the dataset and change all images to size : 224 X 224
images = []
labels = []
print("Loading the model")

for i in range(0, 2):
    imgList = os.listdir('F:/projects/Technocolabs/dataset/' + str(i))
    for j in imgList:
        current_image = cv2.imread('F:/projects/Technocolabs/dataset/' + str(i) + '/' +
                                   str(j))
        current_image = cv2.resize(current_image, (224, 224))
        images.append(current_image)
        labels.append(i)
        
#change images and labels to numpy array
images = np.array(images)
labels = np.array(labels)

plt.imshow(images[2000])

#splitting the data into training set, test set and validation set
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(images, 
                                                            labels, test_size = 0.2)
train_images,  val_images, train_labels, val_labels = train_test_split(train_images, 
                                                            train_labels, test_size=0.2)

#defining callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('val_accuracy')>0.985):
            print("\n 98.5% training is reached, no more epochs ! ")
            self.model.stop_training = True
            
#CNN ARCHITECTURE 
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),
                                                           input_shape=(224, 224, 3),  
                                                           activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Conv2D(64, (3,3),  activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Conv2D(64, (3,3),  activation="relu"),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation="relu"),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                    rotation_range=15,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.2)

callbacks = myCallback()
history = model.fit(datagen.flow(train_images, train_labels, batch_size = 32),
                                  steps_per_epoch=len(train_images)//32,
                                  epochs = 10, 
                                  validation_data = (val_images, val_labels),
                                  validation_steps=len(val_images)//32,
                                  callbacks = [callbacks])

score = model.evaluate(test_images, test_labels, verbose=1)
print('Test Accuracy : ', score[1])

print("[INFO] saving mask detector model...")
model.save("model.h5")

