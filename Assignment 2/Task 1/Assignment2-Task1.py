#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# In[82]:


attributes =  ['vote_result', 'vote1', 'vote2', 'vote3', 'vote4', 'vote5', 'vote6', 'vote7', 'vote8', 'vote9', 'vote10', 
               'vote11', 'vote12', 'vote13', 'vote14', 'vote15', 'vote16']
vote_attributes =  ['vote1', 'vote2', 'vote3', 'vote4', 'vote5', 'vote6', 'vote7', 'vote8', 'vote9', 'vote10', 
               'vote11', 'vote12', 'vote13', 'vote14', 'vote15', 'vote16']
data = pd.read_csv('house-votes-84.txt', header=None)
data.columns = attributes
data.head()


# In[83]:


for index, row in data.iterrows():
    rowData = np.array(row)
    y_count = 0
    n_count = 0
    mark_index = []
    
    for i in range(len(rowData)):
        if rowData[i] == 'y':
            y_count += 1
        elif rowData[i] == 'n':
            n_count += 1
        elif rowData[i] == '?':
            mark_index.append(i)
    
    for i in mark_index:
        if y_count >= n_count:
            rowData[i] = 'y'
        else:
            rowData[i] = 'n'
            
    data.loc[index] = rowData


# In[84]:


data.head()


# In[85]:


class Node():
    def __init__(self, attribute):
        self.attr = attribute
        self.left = None
        self.right = None
        #self.leaf = False
        #self.predict = None


# In[86]:


def calculate_entropy(df, predict_attr):
    # Dataframe and number of republican/democrat examples in the data
    r_df = df[df[predict_attr] == 'republican']
    d_df = df[df[predict_attr] == 'democrat']
    
    # Get number of rows
    r = float(r_df.shape[0])
    d = float(d_df.shape[0])
    
    # Calculate entropy
    if r  == 0 or d == 0:
        entropy = 0
    elif r == d:
        entropy = 1
    else:
        entropy = ((-1*r)/(r + d))*math.log(r/(r+d), 2) + ((-1*d)/(r + d))*math.log(d/(r+d), 2)
    return entropy


# In[87]:


def calculate_entropy_average(df, df_subs, predict_attr):
    # number of test data
    num_data = df.shape[0]
    average = float(0)
    for df_sub in df_subs:
        if df_sub.shape[0] > 1:
            average += float(df_sub.shape[0]/num_data)*calculate_entropy(df_sub, predict_attr)
    return average


# In[89]:


for attr in vote_attributes:
    print("--------" , attr , "---------")
    entropy = calculate_entropy(data, 'vote_result')
    y_data = data[data[attr] == 'y']
    n_data = data[data[attr] == 'n']
    subs = [y_data, n_data]
    entropy_childrens = calculate_entropy_average(data, subs, 'vote_result')
    info_gain = entropy - entropy_childrens
    print(entropy)
    print(entropy_childrens)
    print(info_gain)


# In[ ]:




