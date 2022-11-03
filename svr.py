# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 06:12:51 2022

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

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')


print(svr_reg.predict([[11]]))
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                     
