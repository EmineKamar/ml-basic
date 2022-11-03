# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 06:58:03 2022

@author: kamar
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

veriler = pd.read_csv("Maaslar.csv")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

X = x.values
Y = y.values

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[11]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')

from sklearn.metrics import r2_score

print("R2 Score")
print(r2_score(Y,rf_reg.predict(X)))
