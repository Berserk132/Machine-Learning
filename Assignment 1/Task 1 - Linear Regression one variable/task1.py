#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data = pd.read_csv('house_data.csv', index_col=0)
data.head(5)


# In[3]:


# The Predictor and the target variable
X = data['sqft_living']
Y = data['price']

# convert X and Y to numpy array
X = np.array(X)
Y = np.array(Y)

# Make Scaling for the X data
dataMeanX = np.mean(X)
dataStdX = np.std(X)

X = (X - dataMeanX) / dataStdX 


# # Cost Function Equation
# <div>
# <img src="attachment:O752N.png" width="500"/>
# </div>

# In[4]:


def hypothesis(theta_0, theta_1):
    return theta_0 + theta_1 * X


# In[5]:


# calculate the cost function for the predcitor
def cost_func(hypo, y, m):
    sum_mean = 0
    for i in range(m):
        sum_mean += hypo[i] - y[i]
    sum_mean =  sum_mean / (m)
    return sum_mean


# In[6]:


def square_error(hypo, y, m):
    sum_mean = 0
    for i in range(m):
        sum_mean += (hypo[i] - y[i]) ** 2
    sum_mean = sum_mean / (2 * m)
    return sum_mean


# In[21]:


def gradient_decent(iterations, learning_rate, theta):
    errors = []
    for i in range(iterations):
        hypo = hypothesis(theta[0], theta[1])
        tmp_0 = theta[0] - learning_rate * cost_func(hypo, Y, Y.size)
        tmp_1 = theta[1] - learning_rate * cost_func(hypo * X, Y * X, Y.size)
        theta[0] = tmp_0
        theta[1] = tmp_1
        errors.append(square_error(hypo, Y, Y.size))
    print(f"The Errors of Learning rate {learning_rate} is {errors}")
    print(f"The new {theta[0]} - {theta[1]}")
    
    return errors


# In[24]:


# learning rate of 0.01
iterations = 2000
learning_rate = 0.01
theta = [0, 0]
errors = gradient_decent(iterations, learning_rate, theta)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Real vs predicted values")
plt.scatter(X, Y, color='r')
plt.plot(X, theta[0] + theta[1] * X)
plt.show()


# In[25]:


plt.plot(np.arange(iterations), errors)
plt.xlabel('Iterations')
plt.ylabel('Cost Function')


# In[27]:


x = np.array(data['sqft_living'])
x = (x - dataMeanX) / dataStdX
predicted = theta[0] + theta[1] * x
data['predicted'] = predicted
data.head()


# In[ ]:




