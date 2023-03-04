#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


fb = pd.read_excel(r"D:\Learnerea\Tables\facebook_web_log.xlsx")
fb.head()


# In[3]:


fb = fb.query("action in ('page_load','page_exit')")
fb['page_exit'] = fb.apply(lambda x: x['timestamp'] if x['action']=='page_exit' else None, axis=1)
fb['page_load'] = fb.apply(lambda x: x['timestamp'] if x['action']=='page_load' else None, axis=1)
fb['page_loadN']=fb['page_load'].fillna(method='ffill')
fb = fb.dropna(subset=['page_exit'])


# In[4]:


fb = fb[['user_id','page_exit','page_loadN']]
fb['sec_diff'] = (fb['page_exit']-fb['page_loadN']).dt.total_seconds()
fb


# In[6]:


fb.pivot_table(index='user_id',values='sec_diff',aggfunc='mean').reset_index()

