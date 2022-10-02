#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[144]:


train = pd.read_csv(r'D:\Learnerea\Tables\GiveMeSomeCredit\cs-training.csv').drop(['Unnamed: 0'],axis=1)
test = pd.read_csv(r'D:\Learnerea\Tables\GiveMeSomeCredit\cs-test.csv').drop(['Unnamed: 0'],axis=1)


# In[153]:


train.shape


# In[154]:


test.shape


# In[150]:


train_redup = train.drop_duplicates()


# In[169]:


def findMiss(df):
    return round(df.isnull().sum()/df.shape[0]*100,2)


# In[202]:


train_redup.shape


# In[170]:


findMiss(train_redup)


# In[175]:


train_redup[train_redup.MonthlyIncome.isnull()].describe()


# In[178]:


train_redup['NumberOfDependents'].agg(['mode'])


# In[193]:


fam_miss = train_redup[train_redup.NumberOfDependents.isnull()]
fam_nmiss = train_redup[train_redup.NumberOfDependents.notnull()]


# In[194]:


fam_miss['NumberOfDependents'] = fam_miss['NumberOfDependents'].fillna(0)
fam_miss['MonthlyIncome'] = fam_miss['MonthlyIncome'].fillna(0)


# In[195]:


findMiss(fam_miss)


# In[196]:


findMiss(fam_nmiss)


# In[197]:


fam_nmiss['MonthlyIncome'].agg(['mean','median','min'])


# In[198]:


fam_nmiss['MonthlyIncome'] = fam_nmiss['MonthlyIncome'].fillna(fam_nmiss['MonthlyIncome'].median())


# In[199]:


findMiss(fam_nmiss)


# In[200]:


filled_train = fam_nmiss.append(fam_miss)


# In[203]:


findMiss(filled_train)


# In[204]:


filled_train.head()


# In[207]:


filled_train.groupby(['SeriousDlqin2yrs']).size()/filled_train.shape[0]


# In[208]:


filled_train.RevolvingUtilizationOfUnsecuredLines.describe()


# In[218]:


filled_train['RevolvingUtilizationOfUnsecuredLines'].quantile([.99])


# In[237]:


(filled_train[filled_train['RevolvingUtilizationOfUnsecuredLines'] > 10]).describe()


# In[243]:


util_droped = filled_train.drop(filled_train[filled_train['RevolvingUtilizationOfUnsecuredLines'] > 10].index)


# In[249]:


sns.boxplot(util_droped['age'])


# In[251]:


util_droped.groupby(['NumberOfTime30-59DaysPastDueNotWorse']).size()


# In[252]:


util_droped.groupby(['NumberOfTime60-89DaysPastDueNotWorse']).size()


# In[253]:


util_droped.groupby(['NumberOfTimes90DaysLate']).size()


# In[258]:


util_droped[util_droped['NumberOfTimes90DaysLate']>=96].groupby(['SeriousDlqin2yrs']).size()


# In[260]:


util_droped['DebtRatio'].describe()


# In[262]:


sns.kdeplot(util_droped['DebtRatio'])


# In[289]:


util_droped['DebtRatio'].quantile([.975])


# In[293]:


util_droped[util_droped['DebtRatio']>3492][['SeriousDlqin2yrs','MonthlyIncome']].describe()


# In[297]:


temp = util_droped[(util_droped['DebtRatio']>3492) & (util_droped['SeriousDlqin2yrs']==util_droped['MonthlyIncome'])]


# In[300]:


dRatio = util_droped.drop(util_droped[(util_droped['DebtRatio']>3492) & (util_droped['SeriousDlqin2yrs']==util_droped['MonthlyIncome'])].index)


# In[311]:


# pip install Xgboost


# In[327]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


# In[320]:


model = XGBClassifier(tree_method = 'exact')


# In[318]:


x = dRatio.drop(['SeriousDlqin2yrs'],axis=1)
y = dRatio['SeriousDlqin2yrs']


# In[322]:


model.fit(x,y.values.ravel())
y_pred = model.predict(x)


# In[323]:


accuracy_score(y,y_pred)


# In[325]:


cm = confusion_matrix(y,y_pred)


# In[326]:


sns.heatmap(cm,annot=True,fmt='d',cmap='Oranges',linewidths=0.5,linecolor='Black')
plt.xticks(np.arange(2)+.5,['No def','def'])
plt.yticks(np.arange(2)+.5,['No def','def'])
plt.xlabel("predicted")
plt.ylabel("actuals")


# In[328]:


print(classification_report(y,y_pred))

