#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random


# In[82]:

attributes =  ['vote_result', 'vote1', 'vote2', 'vote3', 'vote4', 'vote5', 'vote6', 'vote7', 'vote8', 'vote9', 'vote10', 
               'vote11', 'vote12', 'vote13', 'vote14', 'vote15', 'vote16']
vote_attributes =  ['vote1', 'vote2', 'vote3', 'vote4', 'vote5', 'vote6', 'vote7', 'vote8', 'vote9', 'vote10', 
               'vote11', 'vote12', 'vote13', 'vote14', 'vote15', 'vote16']
#data = pd.read_csv('D:\PythonProjects\Machine-Learning\Assignment 2\Task 1\house-votes-84.txt', header=None)
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


def getTestandTrainingData(data, percent):
    dataCopy = data
    testingData = dataCopy.sample(frac = percent/100)
    dataCopy.drop(testingData.index)
    trainingData = dataCopy.sample(frac = (100 - percent)/100)
    return trainingData, testingData





class Node():
    def __init__(self, attribute = None):
        self.attr = attribute
        self.left = None
        self.right = None
        self.leaf = False
        self.predict = None

    def getSize(self):
        if self.left and self.right:
            return 1 + self.left.getSize() + self.right.getSize()
        elif self.left:
            return 1 + self.left.getSize()
        elif self.right:
            return 1 + self.right.getSize()
        else:
            return 1

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


def getNextAttribute(df):
    info_gain_dic = dict()
    for attr in vote_attributes:

        entropy = calculate_entropy(df, 'vote_result')
        y_data = df[df[attr] == 'y']
        n_data = df[df[attr] == 'n']
        subs = [y_data, n_data]
        entropy_childrens = calculate_entropy_average(df, subs, 'vote_result')
        info_gain = entropy - entropy_childrens
        info_gain_dic[attr] = info_gain 

    values = list(info_gain_dic.values())
    keys  = list(info_gain_dic.keys())
    maxValue = max(values)
    maxValueIndex = values.index(maxValue)
    maxKey = keys[maxValueIndex]
    #print(maxValue , " " , maxValueIndex, " ", maxKey)

    return maxKey


# In[ ]:
def build_tree(df, predict_attr):
    # Dataframe and number of republican/democrat examples in the data
    r_df = df[df[predict_attr] == 'republican']
    d_df = df[df[predict_attr] == 'democrat']
    #print(r_df.shape[0])
    #print(d_df.shape[0])
    # Get number of rows
    r = float(r_df.shape[0])
    d = float(d_df.shape[0])
    if r == 0 or d == 0:

        leaf = Node(None)
        leaf.leaf = True

        if r == 0:
            leaf.predict = 'democrat'
        
        if d == 0:
            leaf.predict = 'republican'

        

        return leaf

    elif r == 1 and d == 1:

        leaf = Node(None)
        leaf.leaf = True
        leaf.predict = 'None'

        return leaf
        
    else:
        
        bestAttr = getNextAttribute(df)
        tree = Node(bestAttr)
        
        y_data = df[df[bestAttr] == 'y']
        n_data = df[df[bestAttr] == 'n']

        tree.left = build_tree(y_data, predict_attr)
        tree.right = build_tree(n_data, predict_attr)

        return tree




# %%
def predict(node, row_df):

	if node.leaf:
		return node.predict
    
	if row_df[node.attr] == 'y':
        
		return predict(node.left, row_df)

	elif row_df[node.attr] == 'n':
		return predict(node.right, row_df)

#%%

# try different sizes of training dataset
counter = 0
trainingSetSize = 30
SizeList = []
accuracyList = []
for j in range (0,5):
    print( "------------" , '\n\nTraining set size of  ',trainingSetSize, '%',"---------------")
    for i in range(0,5):
        tree = Node()
        trainData, testData = getTestandTrainingData(data, trainingSetSize)
        
        tree = build_tree(trainData, 'vote_result')
        
        for index,row in testData.iterrows():
            prediction = predict(tree, row)
            if prediction == row['vote_result']:
                    counter += 1
        
        treeSize = tree.getSize()
        accuracy = (counter / testData.shape[0]) * 100
        print( "------------" , 'Tree Number ',i+1,"---------------")
        print(accuracy)
        
        print("The Size of the tree = ", treeSize)
        counter = 0
        SizeList.append(treeSize)
        accuracyList.append(accuracy)
        
    print("\nMean of Accurecy = ", sum(accuracyList)/len(accuracyList))
    print("Min of Accurecy = ", min(accuracyList))
    print("Max of Accurecy = ", max(accuracyList))

    print("\nMean of Tree Size = ", sum(SizeList)/len(SizeList))
    print("Min of Tree Size = ",  min(SizeList))
    print("Max of Tree Size = ", max(SizeList))

    counter = 0
    trainingSetSize += 10
    SizeList.clear()
    accuracyList.clear()
