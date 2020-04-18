#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:13:33 2020

@author: amankumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


ds=pd.read_csv('Social_Network_Ads.csv')
X=ds.iloc[:,2:4].values
y=ds.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)



from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
random_classifier=  RandomForestClassifier(n_estimators=10,criterion='gini',random_state=0)
random_classifier.fit(X_train,y_train)
y_pred=random_classifier.predict(X_test)


count=0
for i in range(len(y_pred)):
    if y_pred[i]==y_test[i]:
        count+=1
    
accuracy=count/len(y_pred)

print('Accuracy=',accuracy)    


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, random_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
    
    
'''
source=https://towardsdatascience.com/understanding-random-forest-58381e0602d2
       
       https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
'''
    
