#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:26:49 2020

@author: amankumar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as p


ds=p.read_csv('Data.csv')
indep=ds.iloc[:,:-1].values
dep=ds.iloc[:,-1].values

#Calculating the missing values
from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(indep[:,1:3])
#print(imputer.statistics_)
indep[:,1:3]=imputer.transform(indep[:,1:3])
#print(imputer.fit(indep[:,1:3]))
#print(imputer.transform(indep[:,1:3]))

#categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

label_x=LabelEncoder()
indep[:,0]=label_x.fit_transform(indep[:,0])
onehotenc_x=OneHotEncoder(categorical_features=[0])
indep=onehotenc_x.fit_transform(indep).toarray()
label_y=LabelEncoder()
dep=label_y.fit_transform(dep)


#Splitting Dataset int train and test data

from sklearn.model_selection import train_test_split
indep_train,indep_test,dep_train,dep_test=train_test_split(indep,dep,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
indep_train=sc_x.fit_transform(indep_train)
indep_test=sc_x.transform(indep_test)