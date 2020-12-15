#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[19]:


data = pd.read_csv('house_data.csv', index_col=0)
data.head(5)


# In[26]:


sns.pairplot(data,x_vars=['grade','bathrooms','lat', 'sqft_living', 'view'], y_vars=['price'], height=4, aspect=1)


# In[4]:


def feature_scaling(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    output = (x - x_mean) / x_std
    
    return output


# In[5]:


# The Predictor and the target variable
X1 = data['grade']
X2 = data['bathrooms']
X3 = data['lat']
X4 = data['sqft_living']
X5 = data['view']
Y = data['price']



# convert X and Y to numpy array
X1 = np.array(X1)
X2 = np.array(X2)
X3 = np.array(X3)
X4 = np.array(X4)
X5 = np.array(X5)
Y = np.array(Y)

# Make Scaling for the the features
X1_new = feature_scaling(X1)
X2_new = feature_scaling(X2)
X3_new = feature_scaling(X3)
X4_new = feature_scaling(X4)
X5_new = feature_scaling(X5)

features_list = [X1_new, X2_new, X3_new, X4_new, X5_new]


# # Cost Function Equation
# <div>
# <img src="attachment:O752N.png" width="500"/>
# </div>

# In[6]:


def hypothesis(*args):
    nums = args[0]
    hypo = nums[0]
    for i in range(1, len(nums)):
        hypo += nums[i] * features_list[i-1]
    return hypo


# In[7]:


theta_list = [-1,-5,5,4,3,2]
hypothesis(theta_list)


# In[8]:


# calculate the cost function for the predcitor
def cost_func(hypo, y, m, X):
    sum_mean = 0.0
    for i in range(m):
        if len(X) == 1:
            sum_mean += (hypo[i] - y[i])
        else:
            sum_mean += (hypo[i] - y[i]) * X[i]
    sum_mean =  sum_mean / (m)
    return sum_mean


# In[9]:


def square_error(hypo, y, m):
    sum_mean = 0.0
    for i in range(m):
        sum_mean += (hypo[i] - y[i]) ** 2
    sum_mean = float("{:.2f}".format(sum_mean / (2 * m)))
    return sum_mean


# In[10]:


def gradient_decent(iterations, learning_rate, theta_list):
    tmps = []
    errors = []
    for i in range(iterations):
        hypo = hypothesis(theta_list)
        tmps.append(theta_list[0] - learning_rate * cost_func(hypo, Y, Y.size, [1]))
        for i in range(1, len(theta_list)):
            tmps.append(theta_list[i] - learning_rate * cost_func(hypo , Y, Y.size, features_list[i - 1]))
        theta_list.clear()
        for i in tmps:
            theta_list.append(i)
        tmps.clear()
        errors.append(square_error(hypo, Y, Y.size))
        #print("Error" , error)
        #print(f"The new theta list {theta_list}")
    return errors


# In[11]:


# learning rate of 0.01
iterations = 2000
learning_rate = 0.01
theta_list = [0,0,0,0,0,0]
errors = gradient_decent(iterations, learning_rate, theta_list)


# In[16]:


plt.plot(np.arange(iterations), errors)
plt.xlabel('Iterations')
plt.ylabel('Cost Function')


# In[17]:


x1 = feature_scaling(X1)
x2 = feature_scaling(X2)
x3 = feature_scaling(X3)
x4 = feature_scaling(X4)
x5 = feature_scaling(X5)

features = [x1, x2, x3, x4, x5]
predicted = theta_list[0]

for i in range(1, len(theta_list)):
    predicted += theta_list[i] * features[i - 1]

data['predicted'] = predicted
data.head()


# In[ ]:




