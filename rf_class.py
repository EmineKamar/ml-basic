# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 07:04:20 2022

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


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

###tahmin olasılıkları'proba'

y_proba = rfc.predict_proba(X_test)
print(y_proba)

####ROC 

from sklearn import metrics
fpr, tpr, thold =metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)