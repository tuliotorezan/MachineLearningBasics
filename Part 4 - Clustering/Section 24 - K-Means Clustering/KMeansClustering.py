# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:28:05 2019

@author: Computador
"""

#importing main libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import the mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using ellbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number os clusters')
plt.ylabel('WCSS')
plt.show()

#applying kmeans to the mall dataset with the right number of clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
Y_kmeans = kmeans.fit_predict(X)

#visualising the clusters
plt.scatter(X[Y_kmeans == 0,0], X[Y_kmeans == 0,1], s=100, c = 'red', label = 'Cluster 1')
plt.scatter(X[Y_kmeans == 1,0], X[Y_kmeans == 1,1], s=100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[Y_kmeans == 2,0], X[Y_kmeans == 2,1], s=100, c = 'green', label = 'Cluster 3')
plt.scatter(X[Y_kmeans == 3,0], X[Y_kmeans == 3,1], s=100, c = 'yellow', label = 'Cluster 4')
plt.scatter(X[Y_kmeans == 4,0], X[Y_kmeans == 4,1], s=100, c = 'purple', label = 'Cluster 5')


plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c = 'cyan', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('spending score(1-100)')
plt.legend()
plt.show()



