#!/usr/bin/env python
# coding: utf-8

# In[118]:


my_vocab = ['apple','orange','shimla','banana','maruti','mumbai','china','india','husband'
            ,'wife','brother','sister','red','yellow','computer','mobile','pear','guava']


# In[36]:


import gensim
from sklearn.manifold import TSNE


# In[2]:


import gensim.downloader as api


# In[3]:


glove_model = api.load('glove-wiki-gigaword-300')


# In[130]:


glove_model.most_similar('love',topn=5)


# In[133]:


# husband - man + woman = wife


# In[132]:


glove_model.most_similar(positive= ['woman', 'husband'], negative=['man'],topn=1)


# In[135]:


words = []
vectors = []


for word in my_vocab:
    words.append(word)
    vectors.append(glove_model[word])


# In[138]:


dicts = zip(words,vectors)


# In[139]:


import pandas as pd


# In[141]:


dim_model = TSNE(n_components=2, perplexity=3, init='pca', random_state=45)


# In[143]:


fit_model = dim_model.fit_transform(vectors)


# In[145]:


import matplotlib.pyplot as plt


# In[146]:


fit_model


# In[148]:


x = []
y = []

for i in fit_model:
    x.append(i[0])
    y.append(i[1])


# In[167]:


plt.figure(figsize=(8,8))

for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(words[i], xy=(x[i],y[i]),
                 xytext=(2, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom'
                )


# In[ ]:




