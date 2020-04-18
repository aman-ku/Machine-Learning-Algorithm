#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:27:50 2020

@author: amankumar
"""

import numpy as py
import pandas as pd
import matplotlib.pyplot as plt

ds=pd.read_csv('Mall_Customers.csv')
X=ds.iloc[:,3:5].values

#Using Dendrogram to find prefect cluster
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.xlabel('Customer')
plt.ylabel('Distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
pred=hc.fit_predict(X)

#Plot
plt.scatter(X[pred==0, 0], X[pred==0, 1], s=50,color='red')
plt.scatter(X[pred==1, 0], X[pred==1, 1], s=50,color='blue')
plt.scatter(X[pred==2, 0], X[pred==2, 1], s=50,color='green')
plt.scatter(X[pred==3, 0], X[pred==3, 1], s=50,color='purple')
plt.scatter(X[pred==4, 0], X[pred==4, 1], s=50,color='orange')
plt.show()


'''
source-https://towardsdatascience.com/machine-learning-algorithms-part-12-hierarchical-agglomerative-clustering-example-in-python-1e18e0075019
        https://towardsdatascience.com/understanding-the-concept-of-hierarchical-clustering-technique-c6e8243758ec

'''
