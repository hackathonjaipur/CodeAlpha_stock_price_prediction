#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[7]:


df=pd.read_csv(r'google.csv.csv')

df.head()


# In[8]:


df.shape


# In[9]:


df.isnull().sum()


# In[10]:


df.dtypes


# In[12]:


df['CLOSE'].plot()


# In[17]:


# spliiting data 
x=df[['OPEN','HIGH','LOW','VOLUME']].values

y=df['CLOSE'].values


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.2,random_state=0)


# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import statsmodels.api as sm

regressor =LinearRegression()
model = regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)


# In[28]:


predicted=regressor.predict(x_test)
dframe=pd.DataFrame(y_test,predicted)
dfr = pd.DataFrame({'Actual_Price': y_test,'Predicted_price': predicted})
print(dfr)


# In[30]:


plt.figure(figsize=(8,8))
plt.ylabel('Close_price', fontsize=16)
plt.plot(dfr)
plt.legend(['Actual_price','predicted_price'])
plt.show()


# In[31]:


graph = dfr.head(15)
graph.plot(kind='bar')


# In[ ]:




