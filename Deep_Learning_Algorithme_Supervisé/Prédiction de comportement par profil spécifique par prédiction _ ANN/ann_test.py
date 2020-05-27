# -*- coding: utf-8 -*-

# Churn Modeling
# Lib import
#from theano import *
#from keras import *
#import tensor as tf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


#Préparation des données
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:12].values
y = dataset.iloc[:, 13].values

# Encoding the Independent Variable


labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X[:,1:]  
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Partie 2
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialisation 
classifier = Sequential()
#Ajouter les couches d'entrées et couche cachée
classifier.add(Dense(units=8, activation="relu", 
                     kernel_initializer="uniform", input_dim=11))
classifier.add(Dropout(rate=0.1))

#Ajout d'une deuxième couche cachée
classifier.add(Dense(units=8, activation="relu", 
                     kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))

# Ajout la couche de sortie
classifier.add(Dense(units=1, activation="sigmoid", 
                     kernel_initializer="uniform"))

classifier.compile(optimizer="adam",loss="binary_crossentropy",
                   metrics=["accuracy"])

# Train the neural network
classifier.fit(X_train, y_train, batch_size=10, epochs=100, verbose=1, validation_data=(X_test, y_test))

#Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred2 = (y_pred > 0.5)


# New customer
new_prediction = classifier.predict(sc.transform(np.array([[1,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction2 = (new_prediction > 0.5)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, activation="relu", 
                     kernel_initializer="uniform", input_dim=11))
    classifier.add(Dense(units=6, activation="relu", 
                     kernel_initializer="uniform"))
    classifier.add(Dense(units=1, activation="sigmoid", 
                     kernel_initializer="uniform"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",
                   metrics=["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier,batch_size=10, epochs=100)
precision = cross_val_score(estimator=classifier, X=X_train, y=y_train,cv=10)

moyenne = precision.mean()
ecart_type = precision.std()

#Partie 4 : Ajuster le réseau
#Modules
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, activation="relu", 
                     kernel_initializer="uniform", input_dim=11))
    classifier.add(Dense(units=6, activation="relu", 
                     kernel_initializer="uniform"))
    classifier.add(Dense(units=1, activation="sigmoid", 
                     kernel_initializer="uniform"))
    classifier.compile(optimizer=optimizer,loss="binary_crossentropy",
                   metrics=["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {"batch_size": [25,32],
            "epochs": [100, 500],
             "optimizer": ["adam", "rmsprop"]}
grid_search= GridSearchCV(estimator=classifier,
                          param_grid=parameters,
                          scoring="accuracy",
                          cv=10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_precision=grid_search.best_score_


