#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:29:48 2020

@author: amankumar
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ds=pd.read_csv('Social_Network_Ads.csv')
X=ds.iloc[:,2:4].values
y=ds.iloc[:,4].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#X_train=(X_train-X.min())/(X.max()-X.min())        #scaling



from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
X_test=scale.fit_transform(X_test)

#X_train=X_train-X_train.mean()        #scaling


#using gradient descent algo
'''

m=len(X_train)
x0=np.ones(m).reshape(-1,1)
X_train=np.append(x0,X_train,axis=1)

n=len(X_test)
x01=np.ones(n).reshape(-1,1)
X_test=np.append(x01,X_test,axis=1)

B=np.array([0,0,0])
L=0.0001
epochs=10000
    
for i in range(epochs):
    g=X_test.dot(B)
    h=1/(1+np.exp(-g))
    gradient=-2*(X_test.T.dot((y_test-h)*h*(1-h)))
    B=B-L*gradient
    
h_pred=B[0]*X_test[:,0]+B[1]*X_test[:,1]+B[2]*X_test[:,2]

pred_train=1/(1+np.exp(-h_pred))
 

#pred_train=pred_train.reshape(-1,1)
#pred_train=scale2.inverse_transform(pred_train)


for j in range(len(pred_train)):
    if np.round_(pred_train[j],1)<0.5:
        pred_train[j]=0
    elif np.round_(pred_train[j],1)>=0.5:
        pred_train[j]=1


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred_train)


accuracy=0
for i in range(len(pred_train)):
    if pred_train[i]==y_test[i]:
        accuracy+=1
print(f"Accuracy={accuracy/len(pred_train)}") 
'''


#using sklearn library

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(X_train,y_train)

pred_train=lr.predict(X_test)
accuracy=0
for i in range(len(pred_train)):
    if pred_train[i]==y_test[i]:
        accuracy+=1
print(f"Accuracy={accuracy/len(pred_train)}") 




# Visualising the Training set results

'''
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
'''



    
      

'''
Source= https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
   
      https://towardsdatascience.com/logistic-regression-explained-and-implemented-in-python-880955306060

	https://medium.com/mathematics-behind-optimization-of-cost-function/derivative-of-log-loss-function-for-logistic-regression-9b832f025c2d
'''



