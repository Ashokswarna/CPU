
# coding: utf-8

# In[3]:

import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages
get_ipython().magic('matplotlib inline')
from sklearn.metrics import mean_squared_error
import math
from math import sqrt
import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
df = pd.read_csv('C:\\Python Library\\RMG Data\\RMG New Data\\CPU Utilization every 10 minutes from 1stFeb to 31st March.csv')
df_new = pd.read_csv('C:\\Python Library\\RMG Data\\RMG New Data\\New Data - CPU.csv')
df = df.append(df_new, ignore_index = True)


# In[4]:

df['timestamp'] = df[['TheDate','TIME_OF_DAY']].apply(lambda x: ' '.join(x), axis=1)
df.head()


# In[3]:

df['timestamp'] = pd.to_datetime(df['timestamp'], format = '%m/%d/%Y %H:%M:%S')
df = df.drop_duplicates(subset=['timestamp'], keep = 'first')
df.reset_index(inplace=True)


# In[4]:

df['Hour'] = df.timestamp.apply(lambda x: x.hour)
df['Hour'] = df['Hour'].apply(lambda x: '{0:0>2}'.format(x))
df['Min'] = df.timestamp.apply(lambda x: x.minute)
df['Min'] = df['Min'].apply(lambda x: '{0:0>2}'.format(x))
df['Month'] = df.timestamp.apply(lambda x: x.month)
df['Month'] = df['Month'].apply(lambda x: '{0:0>2}'.format(x))
df['Day'] = df.timestamp.apply(lambda x: x.day)
df['Day'] = df['Day'].apply(lambda x: '{0:0>2}'.format(x))
df['Year'] = df.timestamp.apply(lambda x: x.year)
df['Weekday'] = df['timestamp'].dt.weekday_name
df['Index'] = df['Year'].astype(str)+'-'+df['Month']+'-'+df['Day']+' '+df['Hour']+' '+df['Min']
df.head()


# In[5]:

df['CPU_Busy'] = df['CPU_USR'] + df['CPU_SYS']
df1 = df[['Index','CPU_Busy']]
df1.head()


# In[6]:

df1['Index'] = pd.to_datetime(df1['Index'], format = '%Y-%m-%d %H %M')
df1['Weekday'] = df1['Index'].dt.weekday_name
df1.head()


# In[7]:

df1 = df1.set_index('Index')
df1 = df1.resample('10Min').interpolate(method='linear')
df1.reset_index(inplace=True)
df1['Weekday'] = df1['Index'].dt.weekday_name
print(df1.count())


# ### Prediction for Monday's

# In[8]:

warnings.filterwarnings("ignore") # specify to ignore warning messages
df_mon = df1[df1['Weekday'] == 'Monday']
del df_mon['Weekday']
del df_mon['Index']
df_mon.reset_index(inplace=True)
del df_mon['index']
length = len(df_mon.index)-1
df_mon.drop(df_mon.index[length], inplace=True)
start = datetime.datetime.strptime("00:00:00",'%H:%M:%S')
time_list = [start + relativedelta(minutes=x*10) for x in range(0,length)]
df_mon['index'] = time_list
df_mon.set_index(['index'], inplace=True)
df_mon.index.name=None


# In[10]:

#Summary Statistics of series
print(df_mon.describe())


# In[11]:

#Time series plot
ax = df_mon.CPU_Busy.plot(figsize=(10,4), title= 'Hourly CPU Utilization Time Series Plot', fontsize=8)
ax.set(xlabel="Days", ylabel="CPU Utilization Percentage")


# In[12]:

decomposition = seasonal_decompose(df_mon.values, freq=72)
fig = plt.figure() 
fig = decomposition.plot()
fig.set_size_inches(12, 5)


# In[13]:

from statsmodels.tsa.stattools import adfuller
result = adfuller(df_mon.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[14]:

fig = plt.figure(figsize=(8,4))
pyplot.figure(figsize = (8,4))
pyplot.subplot(211)
plot_acf(df_mon.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_mon.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[15]:

mod = sm.tsa.statespace.SARIMAX(df_mon['CPU_Busy'],
                                order=(8, 0, 0),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_mon = mod.fit()

print(results_mon.summary())


# In[16]:

results_mon.plot_diagnostics(figsize=(10, 7))
plt.show()


# In[17]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_mon = results_mon.get_forecast(steps=73)
pred_ci_mon = pred_uc_mon.conf_int(alpha = 0.1)
pred_ci_mon = pred_ci_mon.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[18]:

ax = df_mon.CPU_Busy[-73:].plot(label='observed', figsize=(10,4))
pred_uc_mon.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_mon.index,
                pred_ci_mon.iloc[:, 0],
                pred_ci_mon.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('CPU Load % Mondays')

plt.legend()
plt.show()


# In[19]:

pred_ci_mon['forecast'] = (pred_ci_mon['lower_CI'] + pred_ci_mon['upper_CI'])/2

pred_ci_mon.loc[pred_ci_mon['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_mon.loc[pred_ci_mon['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_mon.loc[pred_ci_mon['forecast'] > 100, 'forecast'] = 100
final_pred_mon = pred_ci_mon[1:]
final_pred_mon.head(10)
final_pred_mon.to_csv('C:\\Python Library\\RMG Data\\forcast mon.csv')


# In[20]:

fig = plt.figure(figsize=(12,4))
sns.tsplot([final_pred_mon.upper_CI[1:], final_pred_mon.forecast[1:],
            final_pred_mon.lower_CI[1:]],ci = [0,100], color="indianred")


# In[21]:

CPU_pred = 15
start_time = str('02:00:00')
end_time = str('05:00:00')

mod = df_mon.between_time(start_time,end_time) + CPU_pred
df_mon_new = pd.merge(df_mon, mod, left_index=True, right_index=True, how='outer')
df_mon_new = df_mon_new.fillna(0)
df_mon_new['CPU_Busy'] = df_mon_new[['CPU_Busy_x', 'CPU_Busy_y']].max(axis=1)
del df_mon_new['CPU_Busy_x']
del df_mon_new['CPU_Busy_y']
df_mon_new.head()


# In[22]:

## Augmented Dickey Fuller Test ##
from statsmodels.tsa.stattools import adfuller
result = adfuller(df_mon_new.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[23]:

## Plotting ACF and PACF
fig = plt.figure(figsize=(10,4))
pyplot.figure(figsize = (10,4))
pyplot.subplot(211)
plot_acf(df_mon_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_mon_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[24]:

mod = sm.tsa.statespace.SARIMAX(df_mon_new['CPU_Busy'],
                                order=(8, 0, 0),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_mon_new = mod.fit()

print(results_mon_new.summary())


# In[25]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_mon_new = results_mon_new.get_forecast(steps=73)
pred_ci_mon_new = pred_uc_mon_new.conf_int(alpha = 0.1)
pred_ci_mon_new = pred_ci_mon_new.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[26]:

ax = df_mon_new.CPU_Busy[-73:].plot(label='observed', figsize=(10,5))
pred_uc_mon_new.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_mon_new.index,
                pred_ci_mon_new.iloc[:, 0],
                pred_ci_mon_new.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('New CPU Load % Mondays')

plt.legend()
plt.show()


# In[27]:

pred_ci_mon_new['forecast'] = (pred_ci_mon_new['lower_CI'] + pred_ci_mon_new['upper_CI'])/2

pred_ci_mon_new.loc[pred_ci_mon_new['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_mon_new.loc[pred_ci_mon_new['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_mon_new.loc[pred_ci_mon_new['forecast'] > 100, 'forecast'] = 100
final_pred_mon_new = pred_ci_mon_new[1:]
final_pred_mon_new.head()


# In[28]:

fig = plt.figure(figsize = (12,4))
ax  = fig.add_subplot(111)
ax.plot(final_pred_mon.index, final_pred_mon['forecast'], c='b', label='Base Forecast',linewidth = 3.0)
ax.plot(final_pred_mon_new.index, final_pred_mon_new['forecast'], c='r', label='Expected Shift',linewidth = 2.0)
#ax.plot([0,len(final_pred_mon.index)],[80,80], linewidth=3)

leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# get the lines and texts inside legend box
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts
plt.setp(leg_lines, linewidth=3)
plt.setp(leg_texts, fontsize='medium')
plt.title('Impact Analysis of New Job', fontsize = 'medium')
plt.show()
#fig.savefig('C:/Users/manoj.e.kumar.sharma/Desktop/Graphs/Impact Graphs/Impact Analysis-Mon.png')


# ### Prediction for Tuesday's

# In[29]:

warnings.filterwarnings("ignore") # specify to ignore warning messages
df_tue = df1[df1['Weekday'] == 'Tuesday']
del df_tue['Weekday']
del df_tue['Index']
df_tue.reset_index(inplace=True)
del df_tue['index']
length = len(df_tue.index)-1
df_tue.drop(df_tue.index[length], inplace=True)
start = datetime.datetime.strptime("00:00:00",'%H:%M:%S')
time_list = [start + relativedelta(minutes=x*10) for x in range(0,length)]
df_tue['index'] = time_list
df_tue.set_index(['index'], inplace=True)
df_tue.index.name=None


# In[31]:

#Summary Statistics of series
print(df_tue.describe())


# In[32]:

#Time series plot
ax = df_tue.CPU_Busy.plot(figsize=(10,4), title= 'Hourly CPU Utilization Time Series Plot', fontsize=8)
ax.set(xlabel="Days", ylabel="CPU Utilization Percentage")


# In[33]:

decomposition = seasonal_decompose(df_tue.values, freq=72)
fig = plt.figure() 
fig = decomposition.plot()
fig.set_size_inches(12, 5)


# In[34]:

from statsmodels.tsa.stattools import adfuller
result = adfuller(df_tue.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[35]:

fig = plt.figure(figsize=(8,4))
pyplot.figure(figsize = (8,4))
pyplot.subplot(211)
plot_acf(df_tue.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_tue.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[36]:

mod = sm.tsa.statespace.SARIMAX(df_tue['CPU_Busy'],
                                order=(6, 1, 2),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_tue = mod.fit()

print(results_tue.summary())


# In[37]:

results_tue.plot_diagnostics(figsize=(10, 7))
plt.show()


# In[38]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_tue = results_tue.get_forecast(steps=73)
pred_ci_tue = pred_uc_tue.conf_int(alpha = 0.1)
pred_ci_tue = pred_ci_tue.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[39]:

ax = df_tue.CPU_Busy[-73:].plot(label='observed', figsize=(10,4))
pred_uc_tue.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_tue.index,
                pred_ci_tue.iloc[:, 0],
                pred_ci_tue.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('CPU Load % Tuesdays')

plt.legend()
plt.show()


# In[40]:

pred_ci_tue['forecast'] = (pred_ci_tue['lower_CI'] + pred_ci_tue['upper_CI'])/2

pred_ci_tue.loc[pred_ci_tue['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_tue.loc[pred_ci_tue['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_tue.loc[pred_ci_tue['forecast'] > 100, 'forecast'] = 100
final_pred_tue = pred_ci_tue[1:]
final_pred_tue.head(10)
final_pred_tue.to_csv('C:\\Python Library\\RMG Data\\forcast tue.csv')


# In[41]:

fig = plt.figure(figsize=(12,4))
sns.tsplot([final_pred_tue.upper_CI[1:], final_pred_tue.forecast[1:],
            final_pred_tue.lower_CI[1:]],ci = [0,100], color="indianred")


# In[42]:

CPU_pred = 15
start_time = str('02:00:00')
end_time = str('05:00:00')

mod = df_tue.between_time(start_time,end_time) + CPU_pred
df_tue_new = pd.merge(df_tue, mod, left_index=True, right_index=True, how='outer')
df_tue_new = df_tue_new.fillna(0)
df_tue_new['CPU_Busy'] = df_tue_new[['CPU_Busy_x', 'CPU_Busy_y']].max(axis=1)
del df_tue_new['CPU_Busy_x']
del df_tue_new['CPU_Busy_y']
df_tue_new.head()


# In[43]:

## Augmented Dickey Fuller Test ##
from statsmodels.tsa.stattools import adfuller
result = adfuller(df_tue_new.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[44]:

##  Plotting ACF and PACF  ##
fig = plt.figure(figsize=(10,4))
pyplot.figure(figsize = (10,4))
pyplot.subplot(211)
plot_acf(df_tue_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_tue_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[45]:

mod = sm.tsa.statespace.SARIMAX(df_tue_new['CPU_Busy'],
                                order=(6, 1, 2),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_tue_new = mod.fit()

print(results_tue_new.summary())


# In[46]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_tue_new = results_tue_new.get_forecast(steps=73)
pred_ci_tue_new = pred_uc_tue_new.conf_int(alpha = 0.1)
pred_ci_tue_new = pred_ci_tue_new.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[47]:

ax = df_tue_new.CPU_Busy[-73:].plot(label='observed', figsize=(10,6))
pred_uc_tue_new.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_tue_new.index,
                pred_ci_tue_new.iloc[:, 0],
                pred_ci_tue_new.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('New CPU Load % Tuesday')

plt.legend()
plt.show()


# In[48]:

pred_ci_tue_new['forecast'] = (pred_ci_tue_new['lower_CI'] + pred_ci_tue_new['upper_CI'])/2

pred_ci_tue_new.loc[pred_ci_tue_new['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_tue_new.loc[pred_ci_tue_new['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_tue_new.loc[pred_ci_tue_new['forecast'] > 100, 'forecast'] = 100
final_pred_tue_new = pred_ci_tue_new[1:]
final_pred_tue_new.head()


# In[49]:

fig = plt.figure(figsize = (12,4))
ax  = fig.add_subplot(111)
ax.plot(final_pred_tue.index, final_pred_tue['forecast'], c='b', label='Base Forecast',linewidth = 3.0)
ax.plot(final_pred_tue_new.index, final_pred_tue_new['forecast'], c='r', label='Expected Shift',linewidth = 2.0)
#ax.plot([0,len(final_pred_tue.index)],[80,80], linewidth=3)

leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# get the lines and texts inside legend box
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts
plt.setp(leg_lines, linewidth=3)
plt.setp(leg_texts, fontsize='medium')
plt.title('Impact Analysis of New Job', fontsize = 'medium')
plt.show()
#fig.savefig('C:/Users/manoj.e.kumar.sharma/Desktop/Graphs/Impact Graphs/Impact Analysis-Tue.png')


# ### Prediction for Wednesday's

# In[50]:

warnings.filterwarnings("ignore") # specify to ignore warning messages
df_wed = df1[df1['Weekday'] == 'Wednesday']
del df_wed['Weekday']
del df_wed['Index']
df_wed.reset_index(inplace=True)
del df_wed['index']
length = len(df_wed.index)-1
df_wed.drop(df_wed.index[length], inplace=True)
start = datetime.datetime.strptime("00:00:00",'%H:%M:%S')
time_list = [start + relativedelta(minutes=x*10) for x in range(0,length)]
df_wed['index'] = time_list
df_wed.set_index(['index'], inplace=True)
df_wed.index.name=None


# In[52]:

#Summary Statistics of series
print(df_wed.describe())


# In[53]:

#Time series plot
ax = df_wed.CPU_Busy.plot(figsize=(10,4), title= 'Hourly CPU Utilization Time Series Plot', fontsize=8)
ax.set(xlabel="Days", ylabel="CPU Utilization Percentage")


# In[54]:

decomposition = seasonal_decompose(df_wed.values, freq=72)
fig = plt.figure() 
fig = decomposition.plot()
fig.set_size_inches(12, 5)


# In[55]:

from statsmodels.tsa.stattools import adfuller
result = adfuller(df_wed.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[56]:

fig = plt.figure(figsize=(8,4))
pyplot.figure(figsize = (8,4))
pyplot.subplot(211)
plot_acf(df_wed.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_wed.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[57]:

mod = sm.tsa.statespace.SARIMAX(df_wed['CPU_Busy'],
                                order=(4, 1, 3),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_wed = mod.fit()

print(results_wed.summary())


# In[58]:

results_wed.plot_diagnostics(figsize=(10, 7))
plt.show()


# In[59]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_wed = results_wed.get_forecast(steps=73)
pred_ci_wed = pred_uc_wed.conf_int(alpha = 0.1)
pred_ci_wed = pred_ci_wed.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[60]:

ax = df_wed.CPU_Busy[-73:].plot(label='observed', figsize=(10,4))
pred_uc_wed.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_wed.index,
                pred_ci_wed.iloc[:, 0],
                pred_ci_wed.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('CPU Load % Wednesday')

plt.legend()
plt.show()


# In[61]:

pred_ci_wed['forecast'] = (pred_ci_wed['lower_CI'] + pred_ci_wed['upper_CI'])/2

pred_ci_wed.loc[pred_ci_wed['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_wed.loc[pred_ci_wed['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_wed.loc[pred_ci_wed['forecast'] > 100, 'forecast'] = 100
final_pred_wed = pred_ci_wed[1:]
final_pred_wed.head(10)
final_pred_wed.to_csv('C:\\Python Library\\RMG Data\\forcast wed.csv')


# In[62]:

fig = plt.figure(figsize=(12,4))
sns.tsplot([final_pred_wed.upper_CI[1:], final_pred_wed.forecast[1:],
            final_pred_wed.lower_CI[1:]],ci = [0,100], color="indianred")


# In[63]:

CPU_pred = 15
start_time = str('02:00:00')
end_time = str('05:00:00')

mod = df_wed.between_time(start_time,end_time) + CPU_pred
df_wed_new = pd.merge(df_wed, mod, left_index=True, right_index=True, how='outer')
df_wed_new = df_wed_new.fillna(0)
df_wed_new['CPU_Busy'] = df_wed_new[['CPU_Busy_x', 'CPU_Busy_y']].max(axis=1)
del df_wed_new['CPU_Busy_x']
del df_wed_new['CPU_Busy_y']
df_wed_new.head()


# In[64]:

## Augmented Dickey Fuller Test ##
from statsmodels.tsa.stattools import adfuller
result = adfuller(df_wed_new.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[65]:

##  Plotting ACF and PACF  ##
fig = plt.figure(figsize=(10,4))
pyplot.figure(figsize = (10,4))
pyplot.subplot(211)
plot_acf(df_wed_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_wed_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[66]:

mod = sm.tsa.statespace.SARIMAX(df_wed_new['CPU_Busy'],
                                order=(4, 1, 3),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_wed_new = mod.fit()

print(results_wed_new.summary())


# In[67]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_wed_new = results_wed_new.get_forecast(steps=73)
pred_ci_wed_new = pred_uc_wed_new.conf_int(alpha = 0.1)
pred_ci_wed_new = pred_ci_wed_new.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[68]:

ax = df_wed_new.CPU_Busy[-73:].plot(label='observed', figsize=(10,6))
pred_uc_wed_new.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_wed_new.index,
                pred_ci_wed_new.iloc[:, 0],
                pred_ci_wed_new.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('New CPU Load % Wednesday')

plt.legend()
plt.show()


# In[69]:

pred_ci_wed_new['forecast'] = (pred_ci_wed_new['lower_CI'] + pred_ci_wed_new['upper_CI'])/2

pred_ci_wed_new.loc[pred_ci_wed_new['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_wed_new.loc[pred_ci_wed_new['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_wed_new.loc[pred_ci_wed_new['forecast'] > 100, 'forecast'] = 100
final_pred_wed_new = pred_ci_wed_new[1:]
final_pred_wed_new.head()


# In[70]:

fig = plt.figure(figsize = (12,4))
ax  = fig.add_subplot(111)
ax.plot(final_pred_wed.index, final_pred_wed['forecast'], c='b', label='Base Forecast',linewidth = 3.0)
ax.plot(final_pred_wed_new.index, final_pred_wed_new['forecast'], c='r', label='Expected Shift',linewidth = 2.0)
#ax.plot([0,len(final_pred_wed.index)],[80,80], linewidth=3)

leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# get the lines and texts inside legend box
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts
plt.setp(leg_lines, linewidth=3)
plt.setp(leg_texts, fontsize='medium')
plt.title('Impact Analysis of New Job', fontsize = 'medium')
plt.show()
#fig.savefig('C:/Users/manoj.e.kumar.sharma/Desktop/Graphs/Impact Graphs/Impact Analysis-Wed.png')


# ### Prediction for Thursday's

# In[71]:

warnings.filterwarnings("ignore") # specify to ignore warning messages
df_thu = df1[df1['Weekday'] == 'Thursday']
del df_thu['Weekday']
del df_thu['Index']
df_thu.reset_index(inplace=True)
del df_thu['index']
length = len(df_thu.index)-1
df_thu.drop(df_thu.index[length], inplace=True)
start = datetime.datetime.strptime("00:00:00",'%H:%M:%S')
time_list = [start + relativedelta(minutes=x*10) for x in range(0,length)]
df_thu['index'] = time_list
df_thu.set_index(['index'], inplace=True)
df_thu.index.name=None


# In[73]:

#Summary Statistics of series
print(df_thu.describe())


# In[74]:

#Time series plot
ax = df_thu.CPU_Busy.plot(figsize=(10,4), title= 'Hourly CPU Utilization Time Series Plot', fontsize=8)
ax.set(xlabel="Days", ylabel="CPU Utilization Percentage")


# In[75]:

decomposition = seasonal_decompose(df_thu.values, freq=72)
fig = plt.figure() 
fig = decomposition.plot()
fig.set_size_inches(12, 5)


# In[76]:

from statsmodels.tsa.stattools import adfuller
result = adfuller(df_thu.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[77]:

fig = plt.figure(figsize=(8,4))
pyplot.figure(figsize = (8,4))
pyplot.subplot(211)
plot_acf(df_thu.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_thu.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[78]:

mod = sm.tsa.statespace.SARIMAX(df_thu['CPU_Busy'],
                                order=(0, 2, 1),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_thu = mod.fit()

print(results_thu.summary())


# In[79]:

results_thu.plot_diagnostics(figsize=(10, 7))
plt.show()


# In[80]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_thu = results_thu.get_forecast(steps=73)
pred_ci_thu = pred_uc_thu.conf_int(alpha = 0.1)
pred_ci_thu = pred_ci_thu.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[81]:

ax = df_thu.CPU_Busy[-73:].plot(label='observed', figsize=(10,4))
pred_uc_thu.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_thu.index,
                pred_ci_thu.iloc[:, 0],
                pred_ci_thu.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('CPU Load % Tuesdays')

plt.legend()
plt.show()


# In[82]:

pred_ci_thu['forecast'] = (pred_ci_thu['lower_CI'] + pred_ci_thu['upper_CI'])/2

pred_ci_thu.loc[pred_ci_thu['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_thu.loc[pred_ci_thu['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_thu.loc[pred_ci_thu['forecast'] > 100, 'forecast'] = 100
final_pred_thu = pred_ci_thu[1:]
final_pred_thu.head(10)
final_pred_thu.to_csv('C:\\Python Library\\RMG Data\\forcast thu.csv')


# In[83]:

fig = plt.figure(figsize=(12,4))
sns.tsplot([final_pred_thu.upper_CI[1:], final_pred_thu.forecast[1:],
            final_pred_thu.lower_CI[1:]],ci = [0,100], color="indianred")


# In[84]:

CPU_pred = 15
start_time = str('02:00:00')
end_time = str('05:00:00')

mod = df_thu.between_time(start_time,end_time) + CPU_pred
df_thu_new = pd.merge(df_thu, mod, left_index=True, right_index=True, how='outer')
df_thu_new = df_thu_new.fillna(0)
df_thu_new['CPU_Busy'] = df_thu_new[['CPU_Busy_x', 'CPU_Busy_y']].max(axis=1)
del df_thu_new['CPU_Busy_x']
del df_thu_new['CPU_Busy_y']
df_thu_new.head()


# In[85]:

## Augmented Dickey Fuller Test ##
from statsmodels.tsa.stattools import adfuller
result = adfuller(df_thu_new.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[86]:

##  Plotting ACF and PACF  ##
fig = plt.figure(figsize=(10,4))
pyplot.figure(figsize = (10,4))
pyplot.subplot(211)
plot_acf(df_thu_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_thu_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[87]:

mod = sm.tsa.statespace.SARIMAX(df_thu_new['CPU_Busy'],
                                order=(0, 2, 1),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_thu_new = mod.fit()

print(results_thu_new.summary())


# In[88]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_thu_new = results_thu_new.get_forecast(steps=73)
pred_ci_thu_new = pred_uc_thu_new.conf_int(alpha = 0.1)
pred_ci_thu_new = pred_ci_thu_new.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[89]:

ax = df_thu_new.CPU_Busy[-73:].plot(label='observed', figsize=(10,6))
pred_uc_thu_new.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_thu_new.index,
                pred_ci_thu_new.iloc[:, 0],
                pred_ci_thu_new.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('New CPU Load % Tuesday')

plt.legend()
plt.show()


# In[90]:

pred_ci_thu_new['forecast'] = (pred_ci_thu_new['lower_CI'] + pred_ci_thu_new['upper_CI'])/2

pred_ci_thu_new.loc[pred_ci_thu_new['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_thu_new.loc[pred_ci_thu_new['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_thu_new.loc[pred_ci_thu_new['forecast'] > 100, 'forecast'] = 100
final_pred_thu_new = pred_ci_thu_new[1:]
final_pred_thu_new.head()


# In[91]:

fig = plt.figure(figsize = (12,4))
ax  = fig.add_subplot(111)
ax.plot(final_pred_thu.index, final_pred_thu['forecast'], c='b', label='Base Forecast',linewidth = 3.0)
ax.plot(final_pred_thu_new.index, final_pred_thu_new['forecast'], c='r', label='Expected Shift',linewidth = 2.0)
#ax.plot([0,len(final_pred_thu.index)],[80,80], linewidth=3)

leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# get the lines and texts inside legend box
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts
plt.setp(leg_lines, linewidth=3)
plt.setp(leg_texts, fontsize='medium')
plt.title('Impact Analysis of New Job', fontsize = 'medium')
plt.show()
#fig.savefig('C:/Users/manoj.e.kumar.sharma/Desktop/Graphs/Impact Graphs/Impact Analysis-thu.png')


# ### Prediction for Friday's

# In[92]:

warnings.filterwarnings("ignore") # specify to ignore warning messages
df_fri = df1[df1['Weekday'] == 'Friday']
del df_fri['Weekday']
del df_fri['Index']
df_fri.reset_index(inplace=True)
del df_fri['index']
length = len(df_fri.index)-1
df_fri.drop(df_fri.index[length], inplace=True)
start = datetime.datetime.strptime("00:00:00",'%H:%M:%S')
time_list = [start + relativedelta(minutes=x*10) for x in range(0,length)]
df_fri['index'] = time_list
df_fri.set_index(['index'], inplace=True)
df_fri.index.name=None


# In[94]:

#Summary Statistics of series
print(df_fri.describe())


# In[95]:

#Time series plot
ax = df_fri.CPU_Busy.plot(figsize=(10,4), title= 'Hourly CPU Utilization Time Series Plot', fontsize=8)
ax.set(xlabel="Days", ylabel="CPU Utilization Percentage")


# In[96]:

decomposition = seasonal_decompose(df_fri.values, freq=72)
fig = plt.figure() 
fig = decomposition.plot()
fig.set_size_inches(12, 5)


# In[97]:

from statsmodels.tsa.stattools import adfuller
result = adfuller(df_fri.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[98]:

fig = plt.figure(figsize=(8,4))
pyplot.figure(figsize = (8,4))
pyplot.subplot(211)
plot_acf(df_fri.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_fri.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[99]:

mod = sm.tsa.statespace.SARIMAX(df_fri['CPU_Busy'],
                                order=(3, 1, 3),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_fri = mod.fit()

print(results_fri.summary())


# In[100]:

results_fri.plot_diagnostics(figsize=(10, 7))
plt.show()


# In[101]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_fri = results_fri.get_forecast(steps=73)
pred_ci_fri = pred_uc_fri.conf_int(alpha = 0.1)
pred_ci_fri = pred_ci_fri.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[ ]:

ax = df_fri.CPU_Busy[-73:].plot(label='observed', figsize=(10,4))
pred_uc_fri.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_fri.index,
                pred_ci_fri.iloc[:, 0],
                pred_ci_fri.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('CPU Load % Fridays')

plt.legend()
plt.show()


# In[ ]:

pred_ci_fri['forecast'] = (pred_ci_fri['lower_CI'] + pred_ci_fri['upper_CI'])/2

pred_ci_fri.loc[pred_ci_fri['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_fri.loc[pred_ci_fri['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_fri.loc[pred_ci_fri['forecast'] > 100, 'forecast'] = 100
final_pred_fri = pred_ci_fri[1:]
final_pred_fri.head(10)
final_pred_fri.to_csv('C:\\Python Library\\RMG Data\\forcast fri.csv')


# In[ ]:

fig = plt.figure(figsize=(12,4))
sns.tsplot([final_pred_fri.upper_CI[1:], final_pred_fri.forecast[1:],
            final_pred_fri.lower_CI[1:]],ci = [0,100], color="indianred")


# In[103]:

CPU_pred = 15
start_time = str('02:00:00')
end_time = str('05:00:00')

mod = df_fri.between_time(start_time,end_time) + CPU_pred
df_fri_new = pd.merge(df_fri, mod, left_index=True, right_index=True, how='outer')
df_fri_new = df_fri_new.fillna(0)
df_fri_new['CPU_Busy'] = df_fri_new[['CPU_Busy_x', 'CPU_Busy_y']].max(axis=1)
del df_fri_new['CPU_Busy_x']
del df_fri_new['CPU_Busy_y']
df_fri_new.head()


# In[104]:

## Augmented Dickey Fuller Test ##
from statsmodels.tsa.stattools import adfuller
result = adfuller(df_fri_new.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[105]:

##  Plotting ACF and PACF  ##
fig = plt.figure(figsize=(10,4))
pyplot.figure(figsize = (10,4))
pyplot.subplot(211)
plot_acf(df_fri_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_fri_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[106]:

mod = sm.tsa.statespace.SARIMAX(df_fri_new['CPU_Busy'],
                                order=(3, 1, 3),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_fri_new = mod.fit()

print(results_fri_new.summary())


# In[107]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_fri_new = results_fri_new.get_forecast(steps=73)
pred_ci_fri_new = pred_uc_fri_new.conf_int(alpha = 0.1)
pred_ci_fri_new = pred_ci_fri_new.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[108]:

ax = df_fri_new.CPU_Busy[-73:].plot(label='observed', figsize=(10,6))
pred_uc_fri_new.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_fri_new.index,
                pred_ci_fri_new.iloc[:, 0],
                pred_ci_fri_new.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('New CPU Load % Tuesday')

plt.legend()
plt.show()


# In[109]:

pred_ci_fri_new['forecast'] = (pred_ci_fri_new['lower_CI'] + pred_ci_fri_new['upper_CI'])/2

pred_ci_fri_new.loc[pred_ci_fri_new['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_fri_new.loc[pred_ci_fri_new['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_fri_new.loc[pred_ci_fri_new['forecast'] > 100, 'forecast'] = 100
final_pred_fri_new = pred_ci_fri_new[1:]
final_pred_fri_new.head()


# In[110]:

fig = plt.figure(figsize = (12,4))
ax  = fig.add_subplot(111)
ax.plot(final_pred_fri.index, final_pred_fri['forecast'], c='b', label='Base Forecast',linewidth = 3.0)
ax.plot(final_pred_fri_new.index, final_pred_fri_new['forecast'], c='r', label='Expected Shift',linewidth = 2.0)
#ax.plot([0,len(final_pred_fri.index)],[80,80], linewidth=3)

leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# get the lines and texts inside legend box
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts
plt.setp(leg_lines, linewidth=3)
plt.setp(leg_texts, fontsize='medium')
plt.title('Impact Analysis of New Job', fontsize = 'medium')
plt.show()
#fig.savefig('C:/Users/manoj.e.kumar.sharma/Desktop/Graphs/Impact Graphs/Impact Analysis-fri.png')


# ### Prediction for Saturday's

# In[9]:

warnings.filterwarnings("ignore") # specify to ignore warning messages
df_sat = df1[df1['Weekday'] == 'Saturday']
del df_sat['Weekday']
del df_sat['Index']
df_sat.reset_index(inplace=True)
del df_sat['index']
length = len(df_sat.index)-1
df_sat.drop(df_sat.index[length], inplace=True)
start = datetime.datetime.strptime("00:00:00",'%H:%M:%S')
time_list = [start + relativedelta(minutes=x*10) for x in range(0,length)]
df_sat['index'] = time_list
df_sat.set_index(['index'], inplace=True)
df_sat.index.name=None


# In[12]:

#Summary Statistics of series
print(df_sat.describe())


# In[13]:

#Time series plot
ax = df_sat.CPU_Busy.plot(figsize=(10,4), title= 'Hourly CPU Utilization Time Series Plot', fontsize=8)
ax.set(xlabel="Days", ylabel="CPU Utilization Percentage")


# In[14]:

decomposition = seasonal_decompose(df_sat.values, freq=72)
fig = plt.figure() 
fig = decomposition.plot()
fig.set_size_inches(12, 5)


# In[15]:

from statsmodels.tsa.stattools import adfuller
result = adfuller(df_sat.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[16]:

fig = plt.figure(figsize=(8,4))
pyplot.figure(figsize = (8,4))
pyplot.subplot(211)
plot_acf(df_sat.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_sat.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[68]:

mod = sm.tsa.statespace.SARIMAX(df_sat['CPU_Busy'],
                                order=(3, 1, 3),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_sat = mod.fit()

print(results_sat.summary())


# In[69]:

results_sat.plot_diagnostics(figsize=(10, 7))
plt.show()


# In[70]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_sat = results_sat.get_forecast(steps=73)
pred_ci_sat = pred_uc_sat.conf_int(alpha = 0.1)
pred_ci_sat = pred_ci_sat.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[71]:

ax = df_sat.CPU_Busy[-73:].plot(label='observed', figsize=(10,4))
pred_uc_sat.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_sat.index,
                pred_ci_sat.iloc[:, 0],
                pred_ci_sat.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('CPU Load % Saturdays')

plt.legend()
plt.show()


# In[72]:

pred_ci_sat['forecast'] = (pred_ci_sat['lower_CI'] + pred_ci_sat['upper_CI'])/2

pred_ci_sat.loc[pred_ci_sat['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_sat.loc[pred_ci_sat['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_sat.loc[pred_ci_sat['forecast'] > 100, 'forecast'] = 100
final_pred_sat = pred_ci_sat[1:]
final_pred_sat.head(10)
final_pred_sat.to_csv('C:\\Python Library\\RMG Data\\forcast sat.csv')


# In[73]:

fig = plt.figure(figsize=(12,4))
sns.tsplot([final_pred_sat.upper_CI[1:], final_pred_sat.forecast[1:],
            final_pred_sat.lower_CI[1:]],ci = [0,100], color="indianred")


# In[32]:

CPU_pred = 15
start_time = str('02:00:00')
end_time = str('05:00:00')

mod = df_sat.between_time(start_time,end_time) + CPU_pred
df_sat_new = pd.merge(df_sat, mod, left_index=True, right_index=True, how='outer')
df_sat_new = df_sat_new.fillna(0)
df_sat_new['CPU_Busy'] = df_sat_new[['CPU_Busy_x', 'CPU_Busy_y']].max(axis=1)
del df_sat_new['CPU_Busy_x']
del df_sat_new['CPU_Busy_y']
df_sat_new.head()


# In[33]:

## Augmented Dickey Fuller Test ##
from statsmodels.tsa.stattools import adfuller
result = adfuller(df_sat_new.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[34]:

##  Plotting ACF and PACF  ##
fig = plt.figure(figsize=(10,4))
pyplot.figure(figsize = (10,4))
pyplot.subplot(211)
plot_acf(df_sat_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_sat_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[35]:

mod = sm.tsa.statespace.SARIMAX(df_sat_new['CPU_Busy'],
                                order=(3, 1, 3),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_sat_new = mod.fit()

print(results_sat_new.summary())


# In[36]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_sat_new = results_sat_new.get_forecast(steps=73)
pred_ci_sat_new = pred_uc_sat_new.conf_int(alpha = 0.1)
pred_ci_sat_new = pred_ci_sat_new.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[37]:

ax = df_sat_new.CPU_Busy[-73:].plot(label='observed', figsize=(10,6))
pred_uc_sat_new.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_sat_new.index,
                pred_ci_sat_new.iloc[:, 0],
                pred_ci_sat_new.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('New CPU Load % Tuesday')

plt.legend()
plt.show()


# In[38]:

pred_ci_sat_new['forecast'] = (pred_ci_sat_new['lower_CI'] + pred_ci_sat_new['upper_CI'])/2

pred_ci_sat_new.loc[pred_ci_sat_new['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_sat_new.loc[pred_ci_sat_new['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_sat_new.loc[pred_ci_sat_new['forecast'] > 100, 'forecast'] = 100
final_pred_sat_new = pred_ci_sat_new[1:]
final_pred_sat_new.head()


# In[39]:

fig = plt.figure(figsize = (12,4))
ax  = fig.add_subplot(111)
ax.plot(final_pred_sat.index, final_pred_sat['forecast'], c='b', label='Base Forecast',linewidth = 3.0)
ax.plot(final_pred_sat_new.index, final_pred_sat_new['forecast'], c='r', label='Expected Shift',linewidth = 2.0)
#ax.plot([0,len(final_pred_sat.index)],[80,80], linewidth=3)

leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# get the lines and texts inside legend box
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts
plt.setp(leg_lines, linewidth=3)
plt.setp(leg_texts, fontsize='medium')
plt.title('Impact Analysis of New Job', fontsize = 'medium')
plt.show()
#fig.savefig('C:/Users/manoj.e.kumar.sharma/Desktop/Graphs/Impact Graphs/Impact Analysis-sat.png')


# ### Prediction for Sunday's

# In[40]:

warnings.filterwarnings("ignore") # specify to ignore warning messages
df_sun = df1[df1['Weekday'] == 'Sunday']
del df_sun['Weekday']
del df_sun['Index']
df_sun.reset_index(inplace=True)
del df_sun['index']
length = len(df_sun.index)-1
df_sun.drop(df_sun.index[length], inplace=True)
start = datetime.datetime.strptime("00:00:00",'%H:%M:%S')
time_list = [start + relativedelta(minutes=x*10) for x in range(0,length)]
df_sun['index'] = time_list
df_sun.set_index(['index'], inplace=True)
df_sun.index.name=None


# In[43]:

#Summary Statistics of series
print(df_sun.describe())


# In[44]:

#Time series plot
ax = df_sun.CPU_Busy.plot(figsize=(10,4), title= 'Hourly CPU Utilization Time Series Plot', fontsize=8)
ax.set(xlabel="Days", ylabel="CPU Utilization Percentage")


# In[45]:

decomposition = seasonal_decompose(df_sun.values, freq=72)
fig = plt.figure() 
fig = decomposition.plot()
fig.set_size_inches(12, 5)


# In[46]:

from statsmodels.tsa.stattools import adfuller
result = adfuller(df_sun.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[47]:

fig = plt.figure(figsize=(8,4))
pyplot.figure(figsize = (8,4))
pyplot.subplot(211)
plot_acf(df_sun.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_sun.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[49]:

mod = sm.tsa.statespace.SARIMAX(df_sun['CPU_Busy'],
                                order=(8, 0, 1),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_sun = mod.fit()

print(results_sun.summary())


# In[50]:

results_sun.plot_diagnostics(figsize=(10, 7))
plt.show()


# In[51]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_sun = results_sun.get_forecast(steps=73)
pred_ci_sun = pred_uc_sun.conf_int(alpha = 0.1)
pred_ci_sun = pred_ci_sun.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[52]:

ax = df_sun.CPU_Busy[-73:].plot(label='observed', figsize=(10,4))
pred_uc_sun.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_sun.index,
                pred_ci_sun.iloc[:, 0],
                pred_ci_sun.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('CPU Load % Fridays')

plt.legend()
plt.show()


# In[53]:

pred_ci_sun['forecast'] = (pred_ci_sun['lower_CI'] + pred_ci_sun['upper_CI'])/2

pred_ci_sun.loc[pred_ci_sun['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_sun.loc[pred_ci_sun['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_sun.loc[pred_ci_sun['forecast'] > 100, 'forecast'] = 100
final_pred_sun = pred_ci_sun[1:]
final_pred_sun.head(10)
final_pred_sun.to_csv('C:\\Python Library\\RMG Data\\forcast sun.csv')


# In[54]:

fig = plt.figure(figsize=(12,4))
sns.tsplot([final_pred_sun.upper_CI[1:], final_pred_sun.forecast[1:],
            final_pred_sun.lower_CI[1:]],ci = [0,100], color="indianred")


# In[55]:

CPU_pred = 15
start_time = str('02:00:00')
end_time = str('05:00:00')

mod = df_sun.between_time(start_time,end_time) + CPU_pred
df_sun_new = pd.merge(df_sun, mod, left_index=True, right_index=True, how='outer')
df_sun_new = df_sun_new.fillna(0)
df_sun_new['CPU_Busy'] = df_sun_new[['CPU_Busy_x', 'CPU_Busy_y']].max(axis=1)
del df_sun_new['CPU_Busy_x']
del df_sun_new['CPU_Busy_y']
df_sun_new.head()


# In[56]:

## Augmented Dickey Fuller Test ##
from statsmodels.tsa.stattools import adfuller
result = adfuller(df_sun_new.CPU_Busy)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[57]:

##  Plotting ACF and PACF  ##
fig = plt.figure(figsize=(10,4))
pyplot.figure(figsize = (10,4))
pyplot.subplot(211)
plot_acf(df_sun_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.subplot(212)
plot_pacf(df_sun_new.CPU_Busy, ax=pyplot.gca(),lags = 40)
pyplot.show()


# In[58]:

mod = sm.tsa.statespace.SARIMAX(df_sun_new['CPU_Busy'],
                                order=(3, 1, 3),
                                seasonal_order=(1, 1, 1, 72),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_sun_new = mod.fit()

print(results_sun_new.summary())


# In[59]:

#Producing and Visualizing future forecasts (72 intervals in future)
pred_uc_sun_new = results_sun_new.get_forecast(steps=73)
pred_ci_sun_new = pred_uc_sun_new.conf_int(alpha = 0.1)
pred_ci_sun_new = pred_ci_sun_new.rename(columns = {'lower CPU_Busy':'lower_CI','upper CPU_Busy':'upper_CI'})


# In[60]:

ax = df_sun_new.CPU_Busy[-73:].plot(label='observed', figsize=(10,6))
pred_uc_sun_new.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_sun_new.index,
                pred_ci_sun_new.iloc[:, 0],
                pred_ci_sun_new.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Day')
ax.set_ylabel('New CPU Load % Tuesday')

plt.legend()
plt.show()


# In[61]:

pred_ci_sun_new['forecast'] = (pred_ci_sun_new['lower_CI'] + pred_ci_sun_new['upper_CI'])/2

pred_ci_sun_new.loc[pred_ci_sun_new['lower_CI'] < 0, 'lower_CI'] = 0
pred_ci_sun_new.loc[pred_ci_sun_new['upper_CI'] > 100, 'upper_CI'] = 100
pred_ci_sun_new.loc[pred_ci_sun_new['forecast'] > 100, 'forecast'] = 100
final_pred_sun_new = pred_ci_sun_new[1:]
final_pred_sun_new.head()


# In[62]:

fig = plt.figure(figsize = (12,4))
ax  = fig.add_subplot(111)
ax.plot(final_pred_sun.index, final_pred_sun['forecast'], c='b', label='Base Forecast',linewidth = 3.0)
ax.plot(final_pred_sun_new.index, final_pred_sun_new['forecast'], c='r', label='Expected Shift',linewidth = 2.0)
#ax.plot([0,len(final_pred_sun.index)],[80,80], linewidth=3)

leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# get the lines and texts inside legend box
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts
plt.setp(leg_lines, linewidth=3)
plt.setp(leg_texts, fontsize='medium')
plt.title('Impact Analysis of New Job', fontsize = 'medium')
plt.show()
#fig.savefig('C:/Users/manoj.e.kumar.sharma/Desktop/Graphs/Impact Graphs/Impact Analysis-sun.png')

