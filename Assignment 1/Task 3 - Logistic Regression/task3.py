#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# In[88]:


data_original = pd.read_csv('heart.csv', index_col=0)
data_copy = pd.read_csv('heart.csv', index_col=0)
data_original.head(5)


# In[89]:


def feature_scaling(column):
    
    x_mean = np.mean(column)
    x_std = np.std(column)
    column = column.map(lambda x: (x - x_mean) / x_std)
    return column


# In[90]:


# The Predictor and the target variable
Y = data_copy['target']

data_copy['trestbps'] = feature_scaling(data_copy['trestbps'])
data_copy['chol'] = feature_scaling(data_copy['chol'])
data_copy['thalach'] = feature_scaling(data_copy['thalach'])
data_copy['oldpeak'] = feature_scaling(data_copy['oldpeak'])


x = data_copy[['trestbps', 'chol', 'thalach', 'oldpeak']]
m = x.shape[0]
x = np.concatenate((np.ones((m,1)),x), axis=1) 
n = x.shape[1]
theta = np.random.random(size=(1,n))*0.1
Y = Y.values.reshape(m,1)


# In[91]:


data_copy.head(5)


# In[92]:


def hypothesis(*args):
    theta = args[0]
    hypo = (1 / (1 + np.exp(-np.dot(x, np.transpose(theta)))))
    return hypo


# In[93]:


print(hypothesis(theta))


# In[94]:


def cost_func(hypo, y, m):
    sum = 0
    for i in range(m):
        sum += y[i]*math.log(hypo[i]) + (1-y[i])*math.log(1-hypo[i])
    sum = -sum / m
    return sum


# In[95]:


def gradient_decent(iterations, learning_rate, theta):
    errors = []
    for i in range(iterations):
        hypo = hypothesis(theta)
        theta = theta - (learning_rate/m) * np.dot(np.transpose(hypo - Y) , x)
        error = cost_func(hypo, Y, m)
        errors.append(error[0])
        #print("Error" , error[0])
        #print(f"The new theta list {theta}")
        #print(hypo)
    return errors


# In[96]:


# learning rate of 0.01
iterations = 10000
learning_rate = 0.01
errors = gradient_decent(iterations, learning_rate, theta)


# In[97]:


plt.plot(np.arange(iterations), errors)
plt.xlabel('Iterations')
plt.ylabel('Cost Function')


# In[98]:


def feature_scaling_predict(column):
    
    x_mean = np.mean(column)
    x_std = np.std(column)
    return (column - x_mean) / x_std


# In[99]:


values = data_copy[['trestbps', 'chol', 'thalach', 'oldpeak']]
values = np.concatenate((np.ones((m,1)),values), axis=1) 
predicted = np.dot(values, np.transpose(theta))
for i in range(len(predicted)):
    if predicted[i][0] > 0:
        predicted[i][0] = 1.0
    else:
        predicted[i][0] = 0.0
data_original['predicted'] = predicted
print(data_original)

