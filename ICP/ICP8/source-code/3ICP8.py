# importing libraries
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler  #Given
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# load dataset
dataset = pd.read_csv("breastcancer.csv")#importing dataset
X = dataset.iloc[:, 2:32].values#extracting features
y = dataset.iloc[:, 1].values
print(dataset.iloc[:, 1].value_counts())
lb_enc = LabelEncoder()#encoding data
y = lb_enc.fit_transform(y)#fititng the encoder
sc=StandardScaler()#defining sc as standardscaler
X_scaled = sc.fit_transform(X)#adding X as X_scaled
#splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,test_size=0.25, random_state=0)#defining parameter X as X_scaled
#implementing the model
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, y_train, epochs=100, verbose=0,  initial_epoch=0)
#getting the summary
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, y_test))
