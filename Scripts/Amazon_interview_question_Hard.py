#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
amazon = pd.read_excel(r"D:\Learnerea\Tables\amazon_pct_diff.xlsx")
amazon.head()


# In[14]:


amazon['yearMonth'] = amazon['created_at'].dt.strftime('%Y-%m')
amazon_agg = amazon.pivot_table(index='yearMonth',values='value',aggfunc='sum').reset_index()
amazon_agg['prev_month'] = amazon_agg['value'].shift()
amazon_agg['rev_diff'] = round(((amazon_agg['value'] - amazon_agg['prev_month'])/amazon_agg['prev_month'])*100,2)
amazon_agg

