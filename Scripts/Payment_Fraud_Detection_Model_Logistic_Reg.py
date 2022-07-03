#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[9]:


df = pd.read_csv(r"D:\Learnerea\Tables\PS_20174392719_1491204439457_log.csv")
df.head()


# In[10]:


df.shape


# In[11]:


df3 = df.drop(['nameOrig','nameDest','isFlaggedFraud'],axis=1)


# In[12]:


df3.head()


# In[15]:


df3['type'].unique()


# In[18]:


dummies = pd.get_dummies(df3['type']).drop(['CASH_IN'],axis=1)
dummies


# In[30]:


df4 = pd.concat([df3,dummies],axis=1).drop(['type'],axis=1)
df4


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(df4.drop(['isFraud'],axis=1),df4.isFraud,test_size=0.2,random_state=False)


# In[32]:


X_test.shape


# In[33]:


model = LogisticRegression()


# In[34]:


model.fit(X_train,y_train)


# In[35]:


model.score(X_test,y_test)


# In[39]:


df4['isFraud'].unique()


# In[42]:


df4.groupby('isFraud').sum()


# In[43]:


predict = model.predict(X_test)
predict


# In[45]:


cm = confusion_matrix(y_test,predict)


# In[55]:


sns.heatmap(cm,cmap='Oranges',annot=True,fmt='d',cbar=False,linecolor='Black',linewidths=5)
plt.xticks(np.arange(2)+.5,['No Fraud','Fraud'])
plt.yticks(np.arange(2)+.5,['No Fraud','Fraud'])
plt.xlabel("predicted")
plt.ylabel("actuals")

