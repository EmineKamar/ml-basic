# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 05:59:19 2022

@author: kamar
"""

import pandas as pd
import numpy as np
import  matplotlib.pyplot as  plt

data = pd.read_csv("veriler.csv")

x = data.iloc[:,1:4].values
y = data.iloc[:,4:].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(x[:,1:4])
x[:,1:4]=imputer.transform(x[:,1:4])

                                                  
from sklearn.model_selection import train_test_split
                          
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
print(y_pred)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)