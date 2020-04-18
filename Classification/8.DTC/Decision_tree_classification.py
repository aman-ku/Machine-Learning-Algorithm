#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:56:36 2020

@author: amankumar
"""

import pandas as pd
import numpy as np
df = pd.read_csv('play_golf.csv')
eps = np.finfo(float).eps

def entropy_last(df):
    Class=df.keys()[-1]    #return last column of df(keys() return the different attributes names of df)
    unique_values=df[Class].unique() #return array of diiferent value of last column   
    entropy=0
    for i in unique_values:
        prob=df[Class].value_counts()[i]/len(df[Class]) #calculating the probability
        entropy+=-prob*np.log2(prob)
        print(entropy)
    return entropy    


def attribute_entropy(df,attribute):
    unique_classes=df[attribute].unique()
    Class=df.keys()[-1]
    unique_last=df[Class].unique()
    information=0
    for i in unique_classes:
        entropy=0
        for j in unique_last:
            num=len(df[attribute][df[attribute]==i][df[df.keys()[-1]]==j])
            den=len(df[attribute][df[attribute]==i])
            entropy+=-num/(den+eps)*np.log2(num/den+eps)
        information+=den/len(df)*entropy     
    return abs(information)


def best_for_split(df):
    
    IG=[]
    for key in df.keys()[:-1]:
        IG.append(entropy_last(df)-attribute_entropy(df,key))
        
    return df.keys()[:-1][np.argmax(IG)]

def get_subtable(df,node,value):
    return df[df[node]==value].reset_index(drop=True)  #returning the matrix of any single class(category) of attribute with serial number index


def build_tree(df,tree=None):
    Class=df.keys()[-1]
    node=best_for_split(df)
    attr=df[node].unique()
    if tree is None:
        tree={}
        tree[node]={}
        for value in attr:
            subtable = get_subtable(df,node,value)
            clValue,counts = np.unique(subtable['Play Golf'],return_counts=True)                        
        
            if len(counts)==1:#Checking purity of subset
                tree[node][value] = clValue[0]                                                    
            else:        
                tree[node][value] = build_tree(subtable) #Calling the function recursively 
                   
    return tree

tree=build_tree(df)
import pprint
pprint.pprint(tree)

def predict(inst,tree):
    '''
    Function to predict for any input variable.
    '''
    #Recursively we go through the tree that we built earlier

    for nodes in tree.keys():        
        
        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction


data = {'Outlook':'Sunny','Temperature':'Cool','Humidity':'High','Windy':True}
inst = pd.Series(data)
prediction = predict(inst,tree)
print(prediction)
