#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


def dataPrepAutomation(inputDF):
    new_df = inputDF.drop(columns=['nameOrig','nameDest'],axis=1)
    dummies = pd.get_dummies(inputDF['type'])
    processedDf = pd.concat([new_df.drop('type',axis=1),dummies],axis=1)
    independent = processedDf.drop('isFraud',axis=1)
    dependent = processedDf['isFraud']
    return independent, dependent


# In[5]:


def smoteBalancing(features, targets):
    from imblearn.over_sampling import SMOTE
    smote = SMOTE()
    x_smote, y_smote = smote.fit_resample(features, targets)
    X_train, X_test, y_train, y_test = train_test_split(x_smote,y_smote, test_size=0.30, random_state=True)
    return X_train, X_test, y_train, y_test

