# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 06:21:47 2022

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

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X),color='blue')

print(r_dt.predict([[11]]))
