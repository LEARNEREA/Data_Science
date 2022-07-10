#!/usr/bin/env python
# coding: utf-8

# In[166]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier


# In[104]:


df = pd.read_csv(r"D:\Learnerea\Tables\PS_20174392719_1491204439457_log.csv")
df.head()


# In[105]:


df.groupby('isFlaggedFraud').count()


# In[106]:


df.groupby('type').count()


# In[116]:


df.pivot_table(values='amount',index='type',columns='isFraud',aggfunc='count')


# In[136]:


dfFltrd = df[df.type.isin(['CASH_OUT','TRANSFER'])]
dfFltrd.pivot_table(values='amount',index='type',columns='isFraud',aggfunc='count')


# In[137]:


encoder = LabelEncoder()


# In[138]:


dfFltrd['typeEncoded'] = encoder.fit_transform(dfFltrd['type'])
dfFltrd.pivot_table(values='amount',index='typeEncoded',columns='isFraud',aggfunc='count')


# In[139]:


df2 = dfFltrd.drop(['step','type','nameOrig','nameDest','isFlaggedFraud'],axis=1)


# In[146]:


df2.head()


# In[150]:


X_train, X_test, y_train, y_test = train_test_split(df2.drop(['isFraud'],axis=1),df2.isFraud,test_size=0.2,random_state=False)


# In[151]:


LogReg = LogisticRegression()


# In[152]:


LogReg.fit(X_train,y_train)


# In[154]:


predicted = LogReg.predict(X_test)


# In[155]:


LogReg.score(X_test,y_test)


# In[157]:


cm = confusion_matrix(y_test,predicted)
cm


# In[165]:


sns.heatmap(cm,annot=True,fmt='d',cmap='Oranges',linewidths=0.5,linecolor='Black')
plt.xticks(np.arange(2)+.5,['No Fraud','Fraud'])
plt.yticks(np.arange(2)+.5,['No Fraud','Fraud'])
plt.xlabel("predicted")
plt.ylabel("actuals")


# In[164]:


print(classification_report(y_test,predicted))


# In[167]:


dModel = DecisionTreeClassifier()


# In[168]:


dModel.fit(X_train,y_train)


# In[169]:


dPredicted = dModel.predict(X_test)


# In[170]:


cm = confusion_matrix(y_test,dPredicted)


# In[171]:


sns.heatmap(cm,annot=True,fmt='d',cmap='Oranges',linewidths=0.5,linecolor='Black')
plt.xticks(np.arange(2)+.5,['No Fraud','Fraud'])
plt.yticks(np.arange(2)+.5,['No Fraud','Fraud'])
plt.xlabel("predicted")
plt.ylabel("actuals")


# In[172]:


print(classification_report(y_test,dPredicted))

