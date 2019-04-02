# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:53:00 2019

@author: tuliotorezan
"""

#Data Preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values # : = all 
Y = dataset.iloc[:,2].values



#Here it doesnt make sense to split the dataset, since the dataset is way too small
"""
#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"""



#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = Y.reshape((len(Y), 1))
Y = sc_Y.fit_transform(Y)


#Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)


#Predicting the result
Y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))

sc_Y.inverse_transform(Y_pred)


#Vizualizing the SVR results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary')

