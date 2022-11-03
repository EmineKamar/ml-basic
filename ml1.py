# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#Eksik verilerin ortalamasını alma

veriler = pd.read_csv("veriler.csv")

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4]= imputer.transform(Yas[:,1:4])
print(Yas)


####################

#kategorik veriyi numeric hale getirme
#kategorileri kolon başlığına getirip hangi kategori varsa 1 yoksa 0 olur
#label_encoding

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

#onehotencoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ohe = OneHotEncoder(categorical_features='all')
# ulke = ohe.fit_transform(ulke).toarray()
# print(ulke)

columnTransformer = ColumnTransformer([('ulke', OneHotEncoder(), [0])],     remainder='passthrough')
ulke=columnTransformer.fit_transform(ulke)
ulke = ulke[:,0:]
print(ulke)
#X = np.array(columnTransformer.fit_transform(X), dtype = np.float64)

#############

sonuc = pd.DataFrame(data=ulke, index=range(28), columns=["us","tr","fr"])
print(sonuc)

