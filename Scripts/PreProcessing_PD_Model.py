#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# In[3]:


def ModePreProcessing(df):

# Outliers Treatment
    cr_age_rmvd = df[df['person_age']<=70]
    cr_age_rmvd.reset_index(drop=True, inplace=True)
    person_emp_rmvd = cr_age_rmvd[cr_age_rmvd['person_emp_length']<=47]
    person_emp_rmvd.reset_index(drop=True, inplace=True)
    cr_data = person_emp_rmvd.copy()
    
# Missing Values Treatment
    cr_data.fillna({'loan_int_rate':cr_data['loan_int_rate'].median()},inplace=True)
    cr_data_copy=cr_data.drop('loan_grade',axis=1)
    cr_data_cat_treated = cr_data_copy.copy()
    
# Categorical Variables Treatment
    person_home_ownership = pd.get_dummies(cr_data_cat_treated['person_home_ownership'],drop_first=True).astype(int)
    loan_intent = pd.get_dummies(cr_data_cat_treated['loan_intent'],drop_first=True).astype(int)
    cr_data_cat_treated['cb_person_default_on_file_binary'] = np.where(cr_data_cat_treated['cb_person_default_on_file']=='Y',1,0)
    data_to_scale = cr_data_cat_treated.drop(['person_home_ownership','loan_intent','loan_status','cb_person_default_on_file','cb_person_default_on_file_binary'],axis=1)

# Scaling the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_scale)
    scaled_df = pd.DataFrame(scaled_data,columns=['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
           'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'])
    scaled_data_combined = pd.concat([scaled_df,person_home_ownership,loan_intent],axis=1)
    scaled_data_combined['cb_person_default_on_file'] = cr_data_cat_treated['cb_person_default_on_file_binary']
    scaled_data_combined['loan_status'] = cr_data_cat_treated['loan_status']
    
# Features and Target Creation    
    target = scaled_data_combined['loan_status']
    features = scaled_data_combined.drop('loan_status',axis=1)

# SMOTE Balancing
    smote = SMOTE()
    balanced_features, balanced_target = smote.fit_resample(features,target)
    
# return the final datasets
    return data_to_scale, features, target, balanced_features, balanced_target

