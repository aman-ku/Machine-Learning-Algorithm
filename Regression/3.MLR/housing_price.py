#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:47:00 2020

@author: amankumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds=pd.read_csv('housing_price_test.csv')
X=ds.iloc[:,0:2].values
id=ds['Id'].values
x0=np.ones(len(X))
X=np.array([x0,X[:,0],X[:,1]]).T
'''
y=ds.iloc[:,2].values

m=len(y)
L=0.0001

X=np.array([x0,X[:,0],X[:,1]]).T
B=np.array([0,0,0])
epochs=100000

for i in range(epochs):
    h=X.dot(B)
    loss=h-y
    gradient=X.T.dot(loss)/m
    B=B-L*gradient
 '''
pred=B[0]*X[:,0]+B[1]*X[:,1]+B[2]*X[:,2]
dic={'Id':id,'SalePrice':pred}
df=pd.DataFrame(dic)
df.to_csv('house_predict.csv')


    
