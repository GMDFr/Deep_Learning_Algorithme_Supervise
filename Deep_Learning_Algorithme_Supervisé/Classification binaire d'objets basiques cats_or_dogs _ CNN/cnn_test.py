# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:17:06 2019

@author: guill
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#init
classifier = Sequential()
#Etape1 : Convolution => Filter
classifier.add(Convolution2D(filters=32, kernel_size=3,strides=1
                             ,input_shape=(64,64, 3),
                             activation="relu"))

#Etape2 : Pooling = Max Matrix
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Ajout d'une couche de convolution
classifier.add(Convolution2D(filters=32, kernel_size=3,strides=1
                             ,input_shape=(64,64, 3),
                             activation="relu"))

classifier.add(MaxPooling2D(pool_size=(2,2)))

# Etape 3 : Flattening
classifier.add(Flatten())

#Etape 4 Couche complètement connectée
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

# Compilation
classifier.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=["accuracy"])

#Entraîner le CNN sur les images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=250, # 8000 / 32 = 250
        epochs=25,
        validation_data=test_set,
        validation_steps=63) # 2000 / 32 = 62,5

import numpy as np
from keras.preprocessing import image

test_image = image.load_img("dataset/single_prediction/cat.jpg",
                            target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = "chien"
else:
    prediction = "chat"

'''
test_image2 = image.load_img("dataset/single_prediction/cat_or_dog_2.jpg",
                            target_size=(64, 64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis=0)
result2 = classifier.predict(test_image2)
training_set.class_indices
if result[0][0] == 1:
    prediction2 = "chien"
else:
    prediction2 = "chat"
'''