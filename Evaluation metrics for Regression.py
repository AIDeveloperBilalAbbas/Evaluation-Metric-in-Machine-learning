#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[4]:


y_true = [300, 200, 400]
y_pred = [310, 190, 420]


# In[6]:


mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)


# In[7]:


print("MAE:", mae)
print("MSE:", mse)
print("R^2 Score:", r2)


# In[ ]:




