# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:59:01 2020

@author: Tushar goel
"""

import pandas as pd 
import numpy as np
from sklearn import metrics
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
api_key="Enter your API key"
ts=TimeSeries(key="Enter your API key",output_format='pandas')
data,metadata=ts.get_intraday(symbol="AAPL",interval="1min",outputsize="full")
data.to_csv ("D:\python_idle\stock.csv", index = True, header=True)
df=pd.read_csv("D:\python_idle\stock.csv")
df.drop(columns=["2. high","1. open","3. low","5. volume"],axis=1,inplace=True)
df.rename(columns={"4. close":"close"},inplace=True)
def conve(x):
  from datetime import datetime
  fmt = '%Y-%m-%d %H:%M:%S'
  d1 =datetime.strptime(x,fmt)
  return d1
df["date"]=df["date"].apply(conve)
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x)
y = scaler.fit_transform(y)
x_train, y_train = [], []
for i in range(60,1000):
    x_train.append(x[i-60:i,0])
    y_train.append(y[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
print("hello",x_train)
print("xtrain shape",x_train.shape)
print("ytrain",y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
print("xtrain",x_train)
x_test = []
y_test=[]
for i in range(60,  926):
    x_test.append(x[i-60:i,0])
    y_test.append(y[i,0])
x_test = np.array(x_test)
y_test=np.array(y_test).reshape(-1,1)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=2)
prediction=model.predict(x_test)
prediction = scaler.inverse_transform(prediction)
print(prediction)
y_test = scaler.inverse_transform(y_test)
print("Final rmse value is =",np.sqrt(np.mean((y_test-prediction)**2)))
dg=pd.DataFrame(data=y_test,columns=["y_test"])
dg2=pd.DataFrame(data=y_train,columns=["y_train"])
dg["pre"]=0
dg["pre"]=prediction
plt.plot(dg[['y_test','pre']])