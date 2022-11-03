# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 07:51:12 2022

@author: kamar
"""
#basit lineer regresyon

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# veriler = pd.read_csv("satislar.csv")
# veriler.drop("Unnamed: 0",axis=1,inplace=True)

# aylar = veriler[["Aylar"]]
# satislar=veriler[["Satislar"]]
 
# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

# #from sklearn.preprocessing import StandardScaler

# # sc = StandardScaler()

# # X_train = sc.fit_transform(x_train)
# # X_test = sc.fit_transform(x_test)

# # Y_train= sc.fit_transform(y_train)
# # Y_test= sc.fit_transform(y_test)


# from sklearn.linear_model import LinearRegression
# lr=LinearRegression()

# lr.fit(x_train,y_train)

# tahmin= lr.predict(x_test)


# x_train=np.array(x_train.sort_index())
# y_train = np.array(y_train.sort_index())
# plt.plot(x_train,y_train)

# çoklu lineer regresyon

import pandas as pd
import numpy as np
import  matplotlib.pyplot as  plt

data = pd.read_csv("veriler.csv")

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

yas = data.iloc[:,1:4].values
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])

ulke = data.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(data.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()

#cinsiyet içinde yepılıyor

c = data.iloc[:,-1:].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(data.iloc[:,-1])

ohe = preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()

# c'de değişkenler 0 ve 1 den oluşuyor. Bunlar birbirini tamamlayan şeyler. Bu yüzden burada Dummy Variable uygulayacağız(sadece bir kolonu alacağız)
#sonuc3 de bu işlemler yapılıyor.

sonuc = pd.DataFrame(data=ulke, index=range(28), columns= ['fr', 'tr', 'us'])

sonuc2 = pd.DataFrame(data=yas, index=range(28), columns=['boy', 'kilo', 'yas'])

cinsiyet = data.iloc[:,-1].values

sonuc3 = pd.DataFrame(data=c[:,:1], index=range(28), columns=['cinsiyet'])

s=pd.concat([sonuc,sonuc2], axis=1)
s2=pd.concat([s,sonuc3], axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#çoklu lineeer

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#boy tahmini

boy = s2.iloc[:,3:4].values #y

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag], axis=1) #x

x_train2, x_test2, y_train2, y_test2 = train_test_split(veri,boy,test_size=0.33,random_state=0)


r2= LinearRegression()
r2.fit(x_train2,y_train2)

y_pred2 = r2.predict(x_test2)

#backward elimination (geriye doğru eleme)

import statsmodels.api as sm

#beta0 değerleri (1'lerden oluşuru) ekliyoruz
X = np.append(arr=np.ones((28,1)).astype(int), values=veri, axis=1)

X_l=veri.iloc[:,[0,1,2,3,4,5]].values

X_l =np.array(X_l, dtype=float)
model=sm.OLS(boy,X_l).fit()
print(model.summary())

#P>|t| değerine bajtığımızda en yüksek x5(4.eleman) çıktı. 
#backward elimination'da bir sonraki adımda en yüksek p değeri sistemden çıkarılır.

X_l2=veri.iloc[:,[0,1,2,3,5]].values

X_l2 =np.array(X_l2, dtype=float)
model=sm.OLS(boy,X_l2).fit()
print(model.summary())
#en yüksek 0.006 çıktı p değeri. 0.05 den çok küçük kabul edilebilir isteğe göre.



