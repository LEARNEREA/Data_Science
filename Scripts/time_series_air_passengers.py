#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


sns.get_dataset_names()


# In[3]:


df = sns.load_dataset('flights')
df['yearMonth'] = pd.to_datetime("01-"+df['month'].astype(str)+"-"+df['year'].astype(str))
df.set_index('yearMonth',inplace=True)
df.head()


# In[4]:


plt.figure(figsize=(10,5))
sns.lineplot(data=df,x=df.index,y=df.passengers)


# In[5]:


df['rollMean']  = df.passengers.rolling(window=12).mean()
df['rollStd']  = df.passengers.rolling(window=12).std()


# In[6]:


plt.figure(figsize=(10,5))
sns.lineplot(data=df,x=df.index,y=df.passengers)
sns.lineplot(data=df,x=df.index,y=df.rollMean)
sns.lineplot(data=df,x=df.index,y=df.rollStd)


# In[7]:


from statsmodels.tsa.stattools import adfuller


# In[8]:


adfTest = adfuller(df['passengers'],autolag='AIC',)


# In[9]:


adfTest


# In[10]:


stats = pd.Series(adfTest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
stats


# In[11]:


for key, values in adfTest[4].items():
    print('criticality',key,":",values)


# In[12]:


def test_stationarity(dataFrame, var):
    dataFrame['rollMean']  = dataFrame[var].rolling(window=12).mean()
    dataFrame['rollStd']  = dataFrame[var].rolling(window=12).std()
    
    from statsmodels.tsa.stattools import adfuller
    adfTest = adfuller(dataFrame[var],autolag='AIC')
    stats = pd.Series(adfTest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
    print(stats)
    
    for key, values in adfTest[4].items():
        print('criticality',key,":",values)
        
    sns.lineplot(data=dataFrame,x=dataFrame.index,y=var)
    sns.lineplot(data=dataFrame,x=dataFrame.index,y='rollMean')
    sns.lineplot(data=dataFrame,x=dataFrame.index,y='rollStd')


# In[13]:


air_df = df[['passengers']]
air_df.head()


# In[14]:


# time shift

air_df['shift'] = air_df.passengers.shift()
air_df['shiftDiff'] = air_df['passengers'] - air_df['shift']
air_df.head()


# In[15]:


test_stationarity(air_df.dropna(),'shiftDiff')


# In[16]:


log_df = df[['passengers']]
log_df['log'] = np.log(log_df['passengers'])
log_df.head()


# In[17]:


test_stationarity(log_df,'log')


# In[18]:


sqrt_df = df[['passengers']]
sqrt_df['sqrt'] = np.sqrt(df['passengers'])
sqrt_df.head()


# In[19]:


test_stationarity(sqrt_df,'sqrt')


# In[20]:


cbrt_df = df[['passengers']]
cbrt_df['cbrt'] = np.cbrt(cbrt_df['passengers'])
cbrt_df.head()


# In[21]:


test_stationarity(cbrt_df,'cbrt')


# In[22]:


log_df2 = log_df[['passengers','log']]
log_df2['log_sqrt'] = np.sqrt(log_df['log'])
log_df2['logShiftDiff'] = log_df2['log_sqrt'] - log_df2['log_sqrt'].shift()
log_df2.head()


# In[23]:


test_stationarity(log_df2.dropna(),'logShiftDiff')


# In[24]:


log_shift = df[['passengers']].copy(deep=True)
log_shift['log'] = np.log(log_shift['passengers'])
log_shift['logShift'] = log_shift['log'].shift()
log_shift['logShiftDiff'] = log_shift['log'] - log_shift['logShift']
log_shift.head()


# In[25]:


test_stationarity(log_shift.dropna(),'logShiftDiff')


# # Next - 2

# In[80]:


airP = df[['passengers']].copy(deep=True)
airP['firstDiff'] = airP['passengers'].diff()
airP['Diff12'] = airP['passengers'].diff(12)


# In[81]:


airP.head()


# In[84]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[86]:


plot_pacf(airP['firstDiff'].dropna(),lags=20);


# In[87]:


plot_acf(airP['firstDiff'].dropna(),lags=20);


# In[88]:


# p = 1, q = 3, d =1


# In[94]:


train = airP[:round(len(airP)*70/100)]
test = airP[round(len(airP)*70/100):]
test.head()


# In[100]:


model = ARIMA(train['passengers'],order=(1,1,3))
model_fit = model.fit()
prediction = model_fit.predict(start=test.index[0],end=test.index[-1])
airP['arimaPred'] = prediction
airP.tail()


# In[121]:


airP.dropna()
sns.lineplot(data=airP,x=airP.index,y='passengers')
sns.lineplot(data=airP,x=airP.index,y='arimaPred')


# In[122]:


from sklearn.metrics import mean_squared_error


# In[124]:


np.sqrt(mean_squared_error(test['passengers'],prediction))


# In[126]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[141]:


plot_pacf(airP['Diff12'].dropna(),lags=20);
plot_acf(airP['Diff12'].dropna(),lags=20);


# In[147]:


model = SARIMAX(train['passengers'],order=(1,1,3),seasonal_order=(2,1,2,12))
model_fit = model.fit()
prediction = model_fit.predict(start=test.index[0],end=test.index[-1])
airP['sarimaxPred'] = prediction


# In[196]:


airP.dropna()
sns.lineplot(data=airP,x=airP.index,y='passengers')
sns.lineplot(data=airP,x=airP.index,y='sarimaxPred')
sns.lineplot(data=airP,x=airP.index,y='arimaPred')
# model_fit.predict(start=futureDate.index[0],end=futureDate.index[-1]).plot(color='black')


# In[144]:


np.sqrt(mean_squared_error(test['passengers'],prediction))


# In[188]:


futureDate = pd.DataFrame(pd.date_range(start='1961-01-01', end='1962-12-01',freq='MS'),columns=['Dates'])
futureDate.set_index('Dates',inplace=True)
futureDate.head()


# In[192]:


model_fit.predict(start=futureDate.index[0],end=futureDate.index[-1])


# In[195]:


airP.dropna()
sns.lineplot(data=airP,x=airP.index,y='passengers')
sns.lineplot(data=airP,x=airP.index,y='sarimaxPred')
sns.lineplot(data=airP,x=airP.index,y='arimaPred')
model_fit.predict(start=futureDate.index[0],end=futureDate.index[-1]).plot(color='black')


# # Next - 3

# In[201]:


checkDf = df[['passengers']]
checkDf['diff1'] = checkDf.diff()
# checkDf['diffInv'] = checkDf['diff1'].diffinv()
checkDf.head()

