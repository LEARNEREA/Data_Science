#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

pd.options.display.max_colwidth = 1000


# In[4]:


df = pd.read_excel(r"D:\Learnerea\Tables\google_file_store.xlsx")
df 


# In[12]:


df['bull'] = df['contents'].apply(lambda x:x.count('bull'))
df['bear'] = df['contents'].apply(lambda x:x.count('bear'))
df['word'] = 'netry'
df


# In[16]:


df.pivot_table(values=['bull','bear'],columns='word',aggfunc='sum').reset_index().sort_values(by='netry',ascending=False)

