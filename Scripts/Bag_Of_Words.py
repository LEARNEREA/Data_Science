#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth',0)


# In[2]:


df = pd.read_csv(r"D:\Learnerea\Tables\imdb_master.csv", encoding="ISO-8859-1").drop(['Unnamed: 0',"file"],axis=1)


# In[3]:


display(df.shape)
display(df.head())
display(df.type.value_counts())
display(df.label.value_counts())


# In[4]:


filtered = df.query("label!='unsup'")


# In[5]:


filtered['label'] = filtered['label'].apply(lambda x: 0 if x == "neg" else 1)


# In[6]:


filtered.label.value_counts()


# # Cleaning starts

# In[7]:


filtered['review_lower'] = filtered['review'].str.lower()


# In[8]:


from nltk.corpus import stopwords


# In[9]:


engStopWords = stopwords.words("english")


# In[10]:


filtered['review_no_stopW'] = filtered['review_lower'].apply(lambda x:" ".join(word for word in x.split() if word not in engStopWords))


# In[11]:


# filtered.head(2)


# In[12]:


from nltk.stem import WordNetLemmatizer


# In[13]:


lemm = WordNetLemmatizer()


# In[14]:


filtered['lemmatized_review'] = filtered['review_no_stopW'].apply(lambda x: " ".join(lemm.lemmatize(word) for word in x.split()))


# In[15]:


filtered.head(1)


# In[16]:


filtered = filtered.drop(['review','review_lower','review_no_stopW'],axis=1)


# In[17]:


filtered = filtered.rename({'lemmatized_review':'review'},axis=1)


# In[18]:


filtered.head()


# # Cleaning Ends

# In[19]:


train = filtered.query("type=='train'").drop(['type'],axis=1)
test = filtered.query("type=='test'").drop(['type'],axis=1)


# In[20]:


x_train = train['review'].values
y_train = train['label'].values


# In[21]:


x_test = test['review'].values
y_test = test['label'].values


# In[22]:


from sklearn.feature_extraction.text import CountVectorizer


# In[23]:


x_train[0]


# In[24]:


vector = CountVectorizer()


# In[28]:


trained_vector = vector.transform(x_train)


# In[29]:


trained_vector[50].toarray()[:50]


# In[30]:


vector.get_feature_names()[500:][:4]


# In[31]:


train[train.review.str.contains('20minutes')]


# In[32]:


train.shape


# In[33]:


from sklearn.naive_bayes import MultinomialNB


# In[34]:


model = MultinomialNB()


# In[35]:


model.fit(trained_vector, y_train)


# In[36]:


test_vector = vector.transform(x_test)


# In[37]:


predicted = model.predict(test_vector)


# In[38]:


from sklearn.metrics import classification_report


# In[39]:


print(classification_report(y_test,predicted))


# In[76]:


final_sentiment = pd.DataFrame(zip(x_test,y_test,predicted),columns=['review','act_sent','pred_sent'])
final_sentiment.head(2)


# In[78]:


final_sentiment.query("act_sent != pred_sent").head(2)


# In[85]:


review = np.array(["that movie was aweful"])


# In[86]:


text_to_pred = vector.transform(review)


# In[87]:


model.predict(text_to_pred)


# In[100]:


def sentiment_check(input_text):
    test = np.array(input_text)
    text_to_pred = vector.transform(test)
    predicted_val = model.predict(text_to_pred)
    if predicted_val == 1:
        print("the review is POSITIVE")
    else:
        print("the review is NEGATIVE")


# In[102]:


sentiment_check(['the movie was aweful'])

