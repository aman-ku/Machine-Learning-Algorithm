#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 22:30:34 2020

@author: amankumar
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds=pd.read_csv('Iris.csv')
X=ds.iloc[:,1:5].values
Y=ds.iloc[:,5].values



def distance(x,row):
    distcol=np.array([])
    for i in range(len(x)):
        dist=math.sqrt(sum(pow(x[i]-row,2)))
        #print(dist,i)
        distcol=np.append(distcol,dist)   
    return distcol

def sort(x,k):
    main=np.array([])
    for a in range(len(x)):
        min=a
        for j in range(a+1,len(x)):
            if x[j]<x[min]:
                min=j
        temp=x[a]
        x[a]=x[min]
        x[min]=temp
        main=np.append(main,min)
    main=main[0:k]   
    return main


row=np.array([6.1,2.9,4.7,1.4])
d=distance(X,row)
s=sort(d,5)
count1=0
count2=0
count3=0
for h in range(5):
    if Y[int(s[h])] == 'Iris-setosa':
        count1+=1
    elif Y[int(s[h])] == 'Iris-versicolor':
        count2+=1
    elif Y[int(s[h])] == 'Iris-virginica':
        count3+=1

if count1>count2 and count1>count3:
    print('Iris-setosa')
elif count2>count1 and count2>count3:
    print('Iris-versicolor')
else:
    print('Iris-virginica')
    
        