# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:18:03 2019
@author: GMD
"""


# Recurrent Neural Network
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Jeu d'entrainement
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set= dataset_train[["Open"]].values

# Feature Scaling
#Standardisation OU Normalisation

#Standardisation
# xstand = (x-mean(x))/standard deviation (x)

#Normalisation
# xnorm = (x-min(x))/(max(x)-min(x))

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Création de la structure avec 60 timesteps est la sortie et 1 sortie
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[(i-60):i,0])
    y_train.append(training_set_scaled[i,0])
X_train = np.array(X_train)
y_train = np.array(y_train)    

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Construction du RNN

#Librairies
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialisation
regressor = Sequential()

# 1ère Couche LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True, 
                   input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0,2))

# 2ème Couche LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0,2))

# 3ème Couche LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0,2))

# 4ème Couche LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0,2))

# Couche de Sortie
regressor.add(Dense(units=1))

# Compilation
regressor.compile(optimizer="adam",loss="mean_squared_error")
#keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
regressor.fit(X_train,y_train,epochs=100, batch_size=32)

# Données de 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test[["Open"]].values

#Prédictions pour 2017
dataset_total = pd.concat((dataset_train["Open"],dataset_test["Open"]), 
                           axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs =sc.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[(i-60):i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualisation des résultats
plt.plot(real_stock_price, color="red",
         label="Prix réel de l'action Google")
plt.plot(predicted_stock_price, color="green", 
         label="Prix prédit de l'action Google")
plt.title("Prédiction de l'action Google")
plt.xlabel("Jour")
plt.ylabel("Prix de l'action")
plt.legend()
plt.show()