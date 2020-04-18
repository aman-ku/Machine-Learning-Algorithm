#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:38:47 2020

@author: amankumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds=pd.read_csv('50_Startups.csv')
RD=ds['R&D Spend'].values
admin=ds['Administration'].values
market=ds['Marketing Spend'].values
profit=ds['Profit'].values

m=len(RD)
x0=np.ones(m)
X=np.array([x0,RD,admin,market]).T
B=np.array([0,0,0,0])
Y=np.array(profit)


L=0.0001
epochs=1000
for i in range(epochs):
    h=X.dot(B)
    loss=h-Y
    gradient=X.T.dot(loss)/m
    B=B-L*gradient
    
print(B)
pred=B[0]*X[:,0]+B[1]*X[:,1]+B[2]*X[:,2]+B[3]*X[:,3] 

'''
https://mubaris.com/posts/linear-regression/
'''


