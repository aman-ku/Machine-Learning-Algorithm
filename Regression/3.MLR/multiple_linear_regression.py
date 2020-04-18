#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 11:06:33 2020

@author: amankumar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
 
ds=pd.read_csv('marks.csv')
math=ds['Math'].values
read=ds['Reading'].values
write=ds['Writing'].values


'''fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(math,read,write,color='red')
plt.show()'''

m=len(math)
x0=np.ones(m)
X=np.array([x0,math,read]).T

#initializng cofficient
B=np.array([0,0,0])
Y=np.array(write)
L=0.0001

def cost_function(X,Y,B):
    m=len(Y)
    J=np.sum((X.dot(B)-Y)**2)/(2*m)
    return J

def gradient_descent(X,Y,B,L,iterations):
    cost_history=[0]*iterations
    m=len(Y)
    for i in range(iterations):
        h=X.dot(B)
        loss=h-Y
        gradient=X.T.dot(loss)/m
        B=B-L*gradient
        cost=cost_function(X,Y,B)
        cost_history[i]=cost
    return B,cost_history

latestB,cost_history=gradient_descent(X,Y,B,L,100000)
print(latestB)
print(cost_history[-1])

writing_pred=latestB[0]*X[:,0]+latestB[1]*X[:,1]+latestB[2]*X[:,2]
print(writing_pred)
'''fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(math,read,writing_pred,color='blue')
plt.show()
'''
plt.plot(X,writing_pred,color='red')

