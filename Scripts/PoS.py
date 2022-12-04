#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


sentence = "Tesla is not as great as we think"


# In[3]:


token = nltk.word_tokenize(sentence)


# In[4]:


posd = nltk.pos_tag(token)
posd


# In[5]:


nltk.help.upenn_tagset('VB')


# In[6]:


# for i in range(len(posd)):
#     print(posd[i][0],"==>",posd[i][1],"==>",nltk.help.upenn_tagset(posd[i][1]))

