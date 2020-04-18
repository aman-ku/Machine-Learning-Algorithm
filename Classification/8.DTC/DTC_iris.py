#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:20:28 2020

@author: amankumar
"""

from pprint import pprint
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
ds=pd.read_csv('Iris.csv')
X=ds.iloc[:,1:6].values


'''
import seaborn as sb
sb.lmplot(data=ds,x="PetalWidthCm",y="PetalLengthCm",hue="Species",fit_reg="False",size=6,aspect=1)
'''


from sklearn.model_selection import train_test_split
X_train,X_test=train_test_split(X,test_size=0.2,random_state=0)


def check_purity(data):
    last=data[:,-1]
    unique_classes=np.unique(last)
    if len(unique_classes)==1:
        return True
    else:
        return False

def classify_data(data):
    last=data[:,-1]
    unique_classes,count_unique=np.unique(last,return_counts=True)
    index=count_unique.argmax()
    classification=unique_classes[index]
    return classification

def get_potential_splits(data):
    potential_splits={}
    _, n_column=np.shape(data)
    for column_index in range(n_column-1):
        potential_splits[column_index]=[]
        for value in range (len(data)):
            if value!=0:
                previous_value=data[value-1,column_index]
                current_value=data[value,column_index]
                potential_split=(previous_value+current_value) / 2
                potential_splits[column_index].append(potential_split)
    return potential_splits            
                
def split_data(data,split_column,split_value):
    split_column_value=data[:,split_column]
    data_below=data[split_column_value <= split_value]
    data_above=data[split_column_value > split_value]
    return data_below,data_above

def calculate_entropy(data):
    last=data[:,-1]
    _, counts=np.unique(last,return_counts=True)
    probabilities=counts/counts.sum()
    entropy=sum(probabilities*-np.log2(probabilities))
    return entropy

def calculate_overall_entropy(data_below,data_above):
    n=len(data_above)+len(data_below)
    p_data_below=len(data_above)/n
    p_data_above=len(data_below)/n
    overall_entropy=(p_data_above*calculate_entropy(data_above)+p_data_below*calculate_entropy(data_below))
    return overall_entropy

def determine_best_split(data,potential_splits):
    overall_entropy=10000
    for i in potential_splits:
        for j in potential_splits[i]:
            data_below,data_above=split_data(data,i,j)
            current_overall_entropy=calculate_overall_entropy(data_below,data_above)
            if current_overall_entropy <= overall_entropy:
                overall_entropy=current_overall_entropy
                best_split_column=i
                best_split_value=j
    return best_split_column,best_split_value

def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    if counter==0:
        global COLUMN_HEADER
        COLUMN_HEADER=df.columns
        data=df.values
    else:
        data=df
        
    if (check_purity(data)) or (len(data)<min_samples) or (counter == max_depth):
        classification=classify_data(data)
        return classification
    
    else:
        counter=+1
        potential_split=get_potential_splits(data)
        split_column,split_values=determine_best_split(data,potential_split)
        data_below,data_above=split_data(data,split_column,split_values)
        feature_name=COLUMN_HEADER[split_column]
        question="{} <= {}".format(feature_name,split_values)
        sub_tree={question:[]}
        
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)
        
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree

tree = decision_tree_algorithm(ds, max_depth=3)
pprint(tree)
        
    
    
    
    
    
            
'''
source = https://github.com/SebastianMantey/Decision-Tree-from-Scratch/blob/master/notebooks/Video%2007%20-%20Classification.ipynb
'''
    
    
    
    
    
    
    
    
    
    
    
    
    