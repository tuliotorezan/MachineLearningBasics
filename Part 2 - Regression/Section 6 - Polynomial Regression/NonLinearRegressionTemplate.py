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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""


#Fitting the regression model to the dataset




#Predicting the result
Y_pred = regressor.predict([[6.5]])


#Vizualizing the polynomial results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')


#Vizualizing the polynomial results (higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')













