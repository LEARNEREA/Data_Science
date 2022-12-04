#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy


# In[2]:


nlp = spacy.load("en_core_web_sm")


# In[3]:


news = nlp("TATA acquired BigBasket Ent. for $45 billion")


# In[6]:


for x in news.ents:
    print(x.text, "==>", x.label_,"==>", spacy.explain(x.label_))


# In[7]:


from spacy import displacy


# In[8]:


displacy.render(news, style='ent')

