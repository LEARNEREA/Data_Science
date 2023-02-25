#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


amazon = pd.read_excel(r"D:\Learnerea\Tables\amazon_trans.xlsx")
amazon.head()


# In[4]:


amazon.sort_values(by=['user_id','created_at'],inplace=True)
amazon.head()


# In[8]:


amazon['day_diff']=amazon.groupby('user_id')['created_at'].diff()
amazon.head(10)


# In[9]:


amazon.info()


# In[11]:


amazon[amazon['day_diff']<= pd.Timedelta('7 days')]['user_id'].unique()


# In[ ]:




