#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:22:08 2020

@author: amankumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds=pd.read_csv('Position_Salaries.csv')
X=ds.iloc[:,1].values.reshape(-1,1)
Y=ds.iloc[:,2].values.reshape(-1,1)

m=len(Y)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)
Y=scaler.fit_transform(Y)


X=np.append(arr=np.ones((m,1)).astype(int),values=X,axis=1)
X=np.append(X,pow(X[:,1],2).reshape(-1,1),axis=1)
X=np.append(X,pow(X[:,1],3).reshape(-1,1),axis=1)
B=np.array([0,0,0,0]).reshape(-1,1)
L=0.0001
epochs=100000

for i in range(epochs):
    h=X.dot(B)
    loss=h-Y
    gradient=X.T.dot(loss)/m
    B=B-L*gradient
   
pred=B[0]*X[:,0]+B[1]*X[:,1]+B[2]*X[:,2]+B[3]*X[:,3]
pred=pred.reshape(-1,1)

plt.scatter(X[:,1],Y,color='red')
plt.plot(X[:,1],pred,color='blue')

