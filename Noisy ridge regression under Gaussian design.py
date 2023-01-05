#!/usr/bin/env python
# coding: utf-8

# In[243]:


import math
import numpy as np
import matplotlib.pyplot as plt

#1
def ridge_genvals(n,d,sig):
    X = np.random.normal(loc=0.0, scale=1/math.sqrt(n), size=(n,d)) #nxd
    w0 = np.random.normal(loc=0.0, scale=1/math.sqrt(n), size=(d,1)) #dx1
    e = np.random.normal(loc=0, scale=sig, size=(n,1)) #nx1
    y = np.matmul(X,w0) + e #nx1
    return (y,w0,X)
#2
def estimate_ridge(X,y,lamda):
    n,d = X.shape
    xt = X.transpose() #dxn
    ID = np.identity(d) #dxd Identity matrix
    A = np.linalg.inv(np.matmul(xt,X) + lamda*ID) #dxd intermediate matrix
    w_est = np.matmul(np.matmul(A,xt),y) #dx1
    return w_est


# In[244]:


ridge_genvals(200,100,0.03) #for n=200 and d=100


# In[245]:


#3
def diff_generator(n,d,var):  
    
    y,w0,X = ridge_genvals(n,d,var)
    
    diff = []
    for lamda in np.linspace(0,1,100):
        w_est = estimate_ridge(X,y,lamda)
        diff.append(np.linalg.norm(w_est- w0))
    return diff


# In[246]:


diff = diff_generator(200,100,0.03) #for n=200 and d=100


# In[247]:


diff


# In[248]:


plt.plot(np.linspace(0,1,100), diff)
plt.xlabel("lambda2 -->")
plt.ylabel("Distance between w_est and w0 -->")


# #### In the above plot where n=200, d=100 and for standard deviation = 0.03 we observe the bias-variance tradeoff. This is the same tradeoff that occurs when choosing the value of regularization parameter(lambda2) w.r.t the expression obtained in 3.6.

# In[ ]:




