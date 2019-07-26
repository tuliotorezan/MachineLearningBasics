# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:52:11 2019

@author: Computador
"""

#Apriori


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#data preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
#transforming the matrix into a list of lists for the apriori function
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

#Training the model
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#viewing the results
results = list(rules)