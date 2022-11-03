# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veri = pd.read_csv("odev_tenis.csv")

outlook = veri[["outlook,"]]

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
outlook = le.fit_transform(outlook)

windy = veri[["windy,"]]
windy = le.fit_transform(windy)

play = veri[["play"]]
play=le.fit_transform(play)

V1=veri[["temperature,","humidity,"]]

windy=pd.DataFrame(data=windy, index=(range(14)), columns=["windy"])
outlook=pd.DataFrame(data=outlook, index=(range(14)), columns=["outlook"])
play=pd.DataFrame(data=play, index=(range(14)), columns=["play"])
                 
v2=pd.concat([outlook,windy],axis=1)
x=pd.concat([V1,v2],axis=1)
y=play
veri2=pd.concat([x,y],axis=1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int), values=veri2, axis=1)

X_l=veri2.iloc[:,[0,1,2,3,4]].values

X_l =np.array(X_l, dtype=float)
model=sm.OLS(y,X_l).fit()
print(model.summary())

X_l=veri2.iloc[:,[4]].values

X_l =np.array(X_l, dtype=float)
model=sm.OLS(y,X_l).fit()
print(model.summary())
