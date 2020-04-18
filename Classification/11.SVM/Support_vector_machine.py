#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:13:08 2020

@author: amankumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

ds=pd.read_csv('Social_Network_Ads.csv')
X=ds.iloc[:,2:4].values
y=ds.iloc[:,4].values


for i in range(len(y)):
    if y[i]==0:
        y[i]=-1
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.transform(X_test)



#using gradient descent
L=0.0001
w=np.array([0,0])
x=np.array([X_train[:,0],X_train[:,1]]).T
epochs=1000


for k in range(epochs):
    y_pred=x.dot(w)
    prod=y_train*y_pred
    count=0
    for val in prod:
        if val>=1:
            cost=0
            w=w-L*(2 - 1/epochs * w)
        else:
            cost=1-val
            w=w+L*(x[count].T.dot(y_train[count])-2*1/epochs*w)
        count+=1    
        
pred=w[0]*X_test[:,0]+w[1]*X_test[:,1]


for l in range(len(pred)):
    if pred[l]>=1:
        pred[l]=1
    else:
        pred[l]=-1


accuracy=0
for i in range(len(pred)):
    if pred[i]==y_test[i]:
        accuracy+=1
        
print("Accuracy=",accuracy)        

#using sklearn
'''
from sklearn.svm import SVC
classfier=SVC(kernel='linear',random_state=0)
classfier.fit(X_train,y_train)

pred=classfier.predict(X_test)
accuracy=0
for i in range(len(pred)):
    if pred[i]==y_test[i]:
        accuracy+=1
        
print("Accuracy=",accuracy)

'''        


'''
source-https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
       https://medium.com/deep-math-machine-learning-ai/chapter-3-support-vector-machine-with-math-47d6193c82be
       https://towardsdatascience.com/understanding-support-vector-machine-part-1-lagrange-multipliers-5c24a52ffc5e


'''

    
    
    
    

          