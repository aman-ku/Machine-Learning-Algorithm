#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:03:56 2020

@author: amankumar
"""
'''# Load the data
import pandas as pd
ds=pd.read_csv('Iris.csv')

from sklearn.datasets import load_iris
iris = load_iris()

from matplotlib import pyplot as plt

# The indices of the features that we are plotting
x_index = 4
y_index = 3

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: ds.iloc[int(i)])

#plt.figure(figsize=(5, 4))
plt.scatter(ds.iloc[:, x_index], ds.iloc[:, y_index], c=ds.iloc[:,-1])
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel('petal_width(cm)')
plt.ylabel('petal_length(cm)')

plt.tight_layout()
plt.show()

'''

# Load the data
from sklearn.datasets import load_iris
iris = load_iris()

from matplotlib import pyplot as plt

# The indices of the features that we are plotting
x_index = 3
y_index = 2

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

#plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()





