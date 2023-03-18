#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


fb = pd.read_excel(r"D:\Learnerea\Tables\fb_friend_requests.xlsx").sort_values(['user_id_sender','date'])
fb


# In[3]:


sent = fb.query("action=='sent'")
sent


# In[4]:


accepted = fb.query("action=='accepted'")
accepted


# In[6]:


combined = sent.merge(accepted,on=['user_id_sender','user_id_receiver'],how='left')
combined


# In[14]:


summary = combined.pivot_table(index='date_x',values=['action_x','action_y'],aggfunc='count').reset_index()
summary['acceptance_rate'] = summary.action_y/summary.action_x
summary[['date_x','acceptance_rate']]

