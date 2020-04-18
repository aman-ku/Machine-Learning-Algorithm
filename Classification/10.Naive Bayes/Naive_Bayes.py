#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:37:04 2020

@author: amankumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats as s

ds=pd.read_csv('Social_Network_Ads.csv')
X=ds.iloc[:,2:4].values
y=ds.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.transform(X_test)

#splitting data as 0 and 1
X0=X_train[y_train==0]
X1=X_train[y_train==1]

#calculating probability for every class(0 and 1)
X0p=len(X0)/len(X)
X1p=len(X1)/len(X)



#calculating mean and standard deviation for every column corresponding to 0 and 1
X01m,X01s=np.mean(X0[:,0]),np.std(X0[:,0])
X02m,X02s=np.mean(X0[:,1]),np.std(X0[:,1])
X11m,X11s=np.mean(X1[:,0]),np.std(X1[:,0])
X12m,X12s=np.mean(X1[:,1]),np.std(X1[:,1])




#Formula-1.0 / (sigma * (2.0 * pi)**(1/2)) * exp(-1.0 * (x - mu)**2 / (2.0 * (sigma**2)))
def predict(X_test,X0p,X1p,X01m,X02m,X11m,X12m,X01s,X02s,X11s,X12s):
    pred=np.zeros(len(X_test))
    for i in range(len(X_test)):
        
        p01=1.0 / (X01s * (2.0 * np.pi)**(1/2)) * np.exp(-1.0 * (X_test[i,0] - X01m)**2 / (2.0 * (X01s**2)))       
        p02=1.0 / (X02s * (2.0 * np.pi)**(1/2)) * np.exp(-1.0 * (X_test[i,1] - X02m)**2 / (2.0 * (X02s**2))) 
        p0=X0p*p01*p02
        p11=1.0 / (X11s * (2.0 * np.pi)**(1/2)) * np.exp(-1.0 * (X_test[i,0] - X11m)**2 / (2.0 * (X11s**2))) 
        p12=1.0 / (X12s * (2.0 * np.pi)**(1/2)) * np.exp(-1.0 * (X_test[i,0] - X12m)**2 / (2.0 * (X12s**2))) 
        p1=X1p*p11*p12
        if p0>p1:
            pred[i]=0
        else:
            pred[i]=1
            
    return pred

pred=predict(X_test,X0p,X1p,X01m,X02m,X11m,X12m,X01s,X02s,X11s,X12s)


'''
from sklearn.naive_bayes import GaussianNB
bayes=GaussianNB()
bayes.fit(X_train,y_train)

pred=bayes.predict(X_test)
'''
    

accuracy=0

for i in range(len(pred)):
    if pred[i]==y_test[i]:
        accuracy+=1
    
print("Accuracy=",accuracy)
   


'''
Source- https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
        https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/
        https://towardsdatascience.com/probability-concepts-explained-probability-distributions-introduction-part-3-4a5db81858dc

'''
    
    
        
        

        


