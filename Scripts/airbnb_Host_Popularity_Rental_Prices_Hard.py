#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

pd.options.display.max_colwidth=1000


# In[4]:


ab = pd.read_excel(r"D:\Learnerea\Tables\airbnb_host_searches.xlsx").drop('id',axis=1)
ab.head()


# In[6]:


ab.duplicated().value_counts()


# In[7]:


ab2=ab.drop_duplicates()
ab2.duplicated().value_counts()


# In[9]:


ab2.dtypes


# In[11]:


ab2['host_id'] = str(ab2['price'])+ab2['room_type']+str(ab2['host_since'])+str(ab2['zipcode'])+str(ab2['number_of_reviews'])
ab2.head()


# In[12]:


ab2['host_popularity'] = ab2['number_of_reviews'].apply(lambda x: 'New' if x==0
                                                       else 'Rising' if x>=1 and x<=5
                                                        else 'Trending Up' if x>=6 and x<=15
                                                        else 'Popular' if x>=16 and x<=40
                                                        else 'Hot'
                                                       )


# In[17]:


summary = ab2.pivot_table(index='host_popularity',values='price',aggfunc={'price':['min','mean','max']}).reset_index()
summary.rename(columns={'max':'max_price',
                       'min':'min_price',
                       'mean':'avg_price'})

