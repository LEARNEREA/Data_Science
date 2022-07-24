#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[38]:


perf_data = pd.read_excel(r"D:\Learnerea\Tables\data_decision_tree.xlsx")


# In[39]:


perf_data.head()


# In[40]:


encoder = LabelEncoder()


# In[42]:


perf_data['cert_encoded'] = encoder.fit_transform(perf_data['Certification'])
perf_data['proj_encoded'] = encoder.fit_transform(perf_data['Projects'])
perf_data['supt_encoded'] = encoder.fit_transform(perf_data['Supported_other_projects'])
perf_data_new = perf_data.drop(['Certification','Projects','Supported_other_projects'],axis=1)
perf_data_new.head()


# In[43]:


inputs = perf_data_new.drop(['Hike_ge_20pct'],axis=1)
result = perf_data_new['Hike_ge_20pct']


# In[56]:


inputs_train, inputs_test, result_train, result_test = train_test_split(inputs,result,test_size=0.2,random_state=False)


# In[57]:


myTree = DecisionTreeClassifier()


# In[58]:


myTree.fit(inputs_train,result_train)


# In[63]:


prediction=myTree.predict(inputs_test)


# In[60]:


result_test


# In[61]:


myTree.score(inputs_test,result_test)


# In[67]:


cm = confusion_matrix(result_test,prediction)


# In[70]:


print(classification_report(result_test,prediction))

