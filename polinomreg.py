# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


veriler = pd.read_csv("Maaslar.csv")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x.values,y.values)

plt.scatter(x.values,y.values)
plt.plot(x.values,lin_reg.predict(x.values), color='blue')


#polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)

x_poly=poly_reg.fit_transform(x.values)
print(x_poly)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y.values)

plt.scatter(x.values,y.values)

plt.plot(x.values,lin_reg2.predict(poly_reg.fit_transform(x.values)))

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)

x_poly=poly_reg.fit_transform(x.values)
print(x_poly)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y.values)

plt.scatter(x.values,y.values)

plt.plot(x.values,lin_reg2.predict(poly_reg.fit_transform(x.values)))


#tahmin
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))