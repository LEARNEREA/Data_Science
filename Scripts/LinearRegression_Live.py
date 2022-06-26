#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[28]:


salary = pd.read_excel(r"D:\Learnerea\Tables\Salary.xlsx",sheet_name="Salary")
salary.head()


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(salary[['YearsExperience']],salary.Salary,test_size=0.2,random_state=False)


# In[13]:


y_test.shape


# In[14]:


model = LinearRegression()


# In[15]:


model.fit(X_train,y_train)


# In[16]:


model.predict(X_test)


# In[17]:


y_test


# In[18]:


model.score(X_test,y_test)


# In[21]:


def salary_predict(exp):
    return model.predict([[exp]])


# In[22]:


exp = float(input("Enter your years of exp: "))
print("The salary you can expect is: ", salary_predict(exp))


# In[25]:


to_pred = pd.read_excel(r"D:\Learnerea\Tables\Salary.xlsx",sheet_name="to_predict")


# In[26]:


model.predict(to_pred)


# In[27]:


to_pred['predicted_salary']=model.predict(to_pred)
to_pred

