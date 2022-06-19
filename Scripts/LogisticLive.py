#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


app = pd.read_excel("D:\Learnerea\Tables\loan_app_fraud.xlsx")
app.head()


# In[5]:


sns.scatterplot(data=app, x="Age",y="Salary_KPM",hue="Fraud")


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(app[["Age","Salary_KPM"]],app.Fraud,test_size=0.2, random_state=False)


# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


model = LogisticRegression()


# In[15]:


model.fit(X_train,y_train)


# In[16]:


model.predict(X_test)


# In[17]:


model.score(X_test,y_test)


# In[18]:


def fraud_pred_mode(age,salary):
    if model.predict([[age,salary]])==1:
        return "Fraud"
    else:
        return "Genuine"
fraud_pred_mode(20,90)


# In[20]:


age = int(input("Enter the age of the applicant: "))
salary = int(input("Enter the Salary of the applicant: "))

print("The applicant seems to be ", fraud_pred_mode(age,salary))

