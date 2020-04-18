#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:06:57 2020

@author: amankumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import random as rd

ds=pd.read_csv('Mall_Customers.csv')
X=ds.iloc[:,3:5].values
epochs=100
k=5 #let assume
centroid=np.array([]).reshape(2,0) 

for i in range(k):
    row=rd.randint(0,199)
    centroid=np.c_[centroid,X[row]]
    
output={}    
distance=np.array([]).reshape(200,0)   

for j in range(k):
    tempdist=np.sum((X-centroid[:,j])**2,axis=1)
    distance=np.c_[distance,tempdist]
c=np.argmin(distance,axis=1)+1
    
y={}

for i in range(k):
    y[i+1]=np.array([]).reshape(2,0)


for j in range(200):
    y[c[j]]=np.c_[y[c[j]],X[j]]


for l in range(k):
    y[l+1]=y[l+1].T


for i in range(k):
    centroid[:,i]=np.mean(y[i+1],axis=0)


for i in range(epochs):
    distance=np.array([]).reshape(200,0)
    for j in range(k):
        tempdist=np.sum((X-centroid[:,j])**2,axis=1)
        distance=np.c_[distance,tempdist]
    c=np.argmin(distance,axis=1)+1
    y={}

    for i in range(k):
        y[i+1]=np.array([]).reshape(2,0)
    
    for j in range(200):
        y[c[j]]=np.c_[y[c[j]],X[j]]
    
    for l in range(k):
        y[l+1]=y[l+1].T
    
    for i in range(k):
        centroid[:,i]=np.mean(y[i+1],axis=0)    
    output=y
    
    '''
#Plotting
plt.scatter(X[:,0],X[:,1],c='black')
plt.xlabel('Income')
plt.ylabel('Number of transaction')
plt.legen()
plt.show()    
'''

color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for i in range(k):
    plt.scatter(output[i+1][:,0],output[i+1][:,1],c=color[i],label=labels[i])

plt.scatter(centroid[0,:],centroid[1,:],s=300,c='yellow',label='centroid') 
plt.xlabel('Income')
plt.ylabel('Number of transaction')
plt.legend()
plt.show()  




'''
source-https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42
'''
   