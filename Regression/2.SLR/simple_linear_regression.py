#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:13:05 2020

@author: amankumar
"""

import numpy as np
import pandas as p
import matplotlib.pyplot as plt

#importing data sets
ds=p.read_csv('Salary_Data.csv')
indep=ds.iloc[:,0].values
dep=ds.iloc[:,-1].values

#Since there is no  nan in the data so we dont need to do imputer and also there is no column with integer so we dont need to do Label   

#splitting data into training set and test set
from sklearn.model_selection import  train_test_split
indep_train,indep_test,dep_train,dep_test=train_test_split(indep,dep,test_size=1/3,random_state=0)

#Most of the ML Library takes care of feature scaling so we dnt have do to it for SLR


#Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
first_model=LinearRegression()
indep_train=indep_train.reshape(-1,1)
dep_train=dep_train.reshape(-1,1)
first_model.fit(indep_train,dep_train)

#Now Predicting values on test data
indep_test=indep_test.reshape(-1,1)
predict=first_model.predict(indep_test)

#Plotting Graph
plt.scatter(indep_test,dep_test,color='red')
plt.plot(indep_test,first_model.predict(indep_test),color='Green')
plt.title('Graph for Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


# SLR using Gradient Descent
m=0
c=0
n=len(dep)
L=0.0001 #learning rate
max_iter=100000
for i in range(max_iter):
    dep_pred=m*indep+c
    dm=(-2/n)*sum(indep*(dep-dep_pred))
    dc=(-2/n)*sum(dep-dep_pred)
    m=m-L*dm
    c=c-L*dc
print(m,c)
dep_pred=m*indep+c
plt.scatter(indep,dep)
plt.plot(indep,dep_pred,color='red')
plt.show()


'''
https://mubaris.com/posts/linear-regression/
'''
