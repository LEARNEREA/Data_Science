#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[6]:


ab = pd.read_excel(r"D:\Learnerea\Tables\airbnb_contacts.xlsx")
ab.head()


# In[14]:


ab_agg=ab.pivot_table(index='id_guest',values='n_messages',aggfunc='sum').reset_index().sort_values(by='n_messages',ascending=False)
ab_agg


# In[21]:


ab_agg['rank'] = ab_agg['n_messages'].rank(ascending=False,method='dense')
ab_agg.head(10)

