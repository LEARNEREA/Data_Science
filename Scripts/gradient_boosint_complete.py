#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


# In[2]:


data  = {"Income_K":[75,67,70,90,80]
         ,"Profession":["Salaried","Business","Salaried","Business","Salaried"]
        ,"Loan_Lakhs":[3,2,5,7,4]}

df = pd.DataFrame(data)
df


# In[4]:


le = LabelEncoder()


# In[5]:


df['prof_enc'] = le.fit_transform(df['Profession'])
df


# In[6]:


x = df[['Income_K','prof_enc']]
y = df['Loan_Lakhs']


# In[27]:


model = GradientBoostingRegressor(n_estimators=5)


# In[28]:


model.fit(x,y)


# In[29]:


prediction = model.predict(x)


# In[30]:


sum((y-prediction)**2)/len(y)


# In[32]:


params = {'n_estimators':range(1,200)}
grid = GridSearchCV(param_grid=params,estimator=model,cv=2, scoring='neg_mean_squared_error')


# In[33]:


grid.fit(x,y)


# In[34]:


grid.best_estimator_


# In[35]:


gb = grid.best_estimator_


# In[36]:


gb.fit(x,y)


# In[37]:


preodiction = gb.predict(x)


# In[38]:


sum((y-prediction)**2)/len(y)

