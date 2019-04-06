# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:27:42 2019

@author: tuliotorezan
"""


#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



a=31
b=50
x = np.array(range(a))

y = np.flip(-x)
x= np.concatenate((y,x))
y = b*b - b*b*x*x/900
y = np.sqrt(y)

plt.plot(x, y, color = 'blue')