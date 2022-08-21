#!/usr/bin/env python
# coding: utf-8

# # Environment setup and Log in to Twitter

# In[7]:


import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
import getpass


# In[12]:


my_user = "LearnereaBot"
# my_pass = getpass.getpass()
my_pass = "Jamesbond#0016"

# In[16]:


search_item = "Liz Truss"


# In[9]:


PATH = "C:\Program Files\drivers\chromedriver_103.exe"
driver = webdriver.Chrome(PATH)
driver.get("https://twitter.com/i/flow/login")
# driver.maximize_window()
sleep(3)


# In[10]:


user_id = driver.find_element(By.XPATH,"//input[@type='text']")
user_id.send_keys(my_user)
user_id.send_keys(Keys.ENTER)


# In[13]:


password = driver.find_element(By.XPATH,"//input[@type='password']")
password.send_keys(my_pass)
password.send_keys(Keys.ENTER)


# # Scrape Tweets mentioning about Liz Truss

# In[18]:


search_box = driver.find_element(By.XPATH,"//input[@data-testid='SearchBox_Search_Input']")
search_box.send_keys(search_item)
search_box.send_keys(Keys.ENTER)


# In[24]:


all_tweets = set()


tweets = driver.find_elements(By.XPATH,"//div[@data-testid='tweetText']")
while True:
    for tweet in tweets:
        all_tweets.add(tweet.text)
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    sleep(3)
    tweets = driver.find_elements(By.XPATH,"//div[@data-testid='tweetText']")
    if len(all_tweets)>50:
        break


# In[25]:


all_tweets = list(all_tweets)
all_tweets[0]


# # Cleaning the Tweets

# In[64]:


import pandas as pd
pd.options.display.max_colwidth = 1000
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[66]:


stp_words = stopwords.words('english')
print(stp_words)


# In[56]:


df = pd.DataFrame(all_tweets,columns=['tweets'])
df.head()


# In[60]:


one_tweet=df.iloc[4]['tweets']
one_tweet


# In[86]:


from textblob import TextBlob
from wordcloud import WordCloud

def TweetCleaning(tweet):
    cleanTweet = re.sub(r"@[a-zA-Z0-9]+","",tweet)
    cleanTweet = re.sub(r"#[a-zA-Z0-9\s]+","",cleanTweet)
    cleanTweet = ' '.join(word for word in cleanTweet.split() if word not in stp_words)
    return cleanTweet

def calPolarity(tweet):
    return TextBlob(tweet).sentiment.polarity

def calSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

def segmentation(tweet):
    if tweet > 0:
        return "positive"
    if tweet == 0:
        return "neutral"
    else:
        return "negative"


# In[88]:


df['cleanedTweets'] = df['tweets'].apply(TweetCleaning)
df['tPolarity'] = df['cleanedTweets'].apply(calPolarity)
df['tSubjectivity'] = df['cleanedTweets'].apply(calSubjectivity)
df['segmentation'] = df['tPolarity'].apply(segmentation)
df.head()


# # Analysis and Visualization

# In[91]:


df.pivot_table(index=['segmentation'],aggfunc={'segmentation':'count'})


# In[94]:


# top 3 most positive
df.sort_values(by=['tPolarity'],ascending=False).head(3)


# In[95]:


# top 3 most negative
df.sort_values(by=['tPolarity'],ascending=True).head(3)


# In[116]:


# 3 neutral
df[df.tPolarity==0]


# In[103]:


import matplotlib.pyplot as plt

consolidated = ' '.join(word for word in df['cleanedTweets'])

wordCloud = WordCloud(width=400, height=200, random_state=20, max_font_size=119).generate(consolidated)

plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[104]:


import seaborn as sns


# In[113]:


df.groupby('segmentation').count()


# In[115]:


plt.figure(figsize=(10,5))
sns.set_style("whitegrid")
sns.scatterplot(data=df, x='tPolarity',y='tSubjectivity',s=100,hue='segmentation')


# In[117]:


sns.countplot(data=df,x='segmentation')


# In[118]:


positive = round(len(df[df.segmentation == 'positive'])/len(df)*100,1)
negative = round(len(df[df.segmentation == 'negative'])/len(df)*100,1)
neutral = round(len(df[df.segmentation == 'neutral'])/len(df)*100,1)

responses = [positive, negative, neutral]
responses

response = {'resp': ['mayWin', 'mayLoose', 'notSure'], 'pct':[positive, negative, neutral]}
pd.DataFrame(response)

