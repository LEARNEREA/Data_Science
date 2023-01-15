#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install fasttext


# In[2]:


import fasttext


# In[3]:


fModel = fasttext.load_model(r"C:\Users\smrvr\Downloads\cc.en.300.bin")


# In[6]:


fModel.get_nearest_neighbors("love")


# In[10]:


fModel.get_subwords('leanerea')


# In[12]:


fModel.get_analogies('delhi','noida','tokyo',k=10)


# # Training model on own data

# In[13]:


import pandas as pd
pd.set_option('display.max_colwidth',0)


# In[18]:


df = pd.read_csv(r"D:\Learnerea\Tables\imdb_master.csv", encoding="ISO-8859-1")[['review','label']].query("label!='unsup'")
df.groupby('label').count()


# In[21]:


myText = df.review[5]
myText


# In[22]:


import re


# In[28]:


def preProcessing(text):
    text = re.sub(r'[^\w\s\']',"",text)
    text = re.sub(' +',' ',text).lower()
    return text


# In[29]:


preProcessing(myText)


# In[31]:


df['processed_review'] = df['review'].map(preProcessing)


# In[32]:


df.head(1)


# In[33]:


df['final_data'] = "__label__"+df['label']+" "+df['processed_review']


# In[39]:


df.head(1)


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


train, test = train_test_split(df, test_size=0.3)


# In[38]:


display(train.shape)
display(test.shape)


# In[40]:


train.to_csv(r"D:\Learnerea\Tables\fastText\new\train_reviews.csv",columns=['final_data'],index=False,header=False)
test.to_csv(r"D:\Learnerea\Tables\fastText\new\test_reviews.csv",columns=['final_data'],index=False,header=False)


# In[41]:


myModel = fasttext.train_supervised(input=r"D:\Learnerea\Tables\fastText\new\train_reviews.csv")


# In[42]:


myModel.test(r"D:\Learnerea\Tables\fastText\new\test_reviews.csv")


# In[53]:


myModel.predict("Thanks for sharing this ...it will really prove to be very helpful")

