#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# In[38]:


attributes =  ['vote_result', 'vote1', 'vote2', 'vote3', 'vote4', 'vote5', 'vote6', 'vote7', 'vote8', 'vote9', 'vote10', 
               'vote11', 'vote12', 'vote13', 'vote14', 'vote15', 'vote16']
data = pd.read_csv('house-votes-84.txt', header=None)
data.columns = attributes
data.head()


# In[39]:


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


# In[40]:


data.head()


# In[41]:


class Node():
    def __init__(self, attribute):
        self.attr = attribute
        self.left = None
        self.right = None
        self.leaf = False
        self.predict = None


# In[51]:


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


# In[54]:


entropy = calculate_entropy(data, 'vote_result')
print(entropy)


# In[ ]:




