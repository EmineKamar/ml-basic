# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 17:45:27 2022

@author: kamar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("musteriler.csv")

X = data.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init= 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)


sonuclar=[]
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_) ##WCSS

plt.plot(range(1,11),sonuclar)



kmeans = KMeans(n_clusters = 4, init= 'k-means++',random_state=123)
y_pred1 = kmeans.fit_predict(X)
print(y_pred1)
plt.scatter(X[y_pred1==0,0],X[y_pred1==0,1],s=100,c='red')
plt.scatter(X[y_pred1==1,0],X[y_pred1==1,1],s=100,c='blue')
plt.scatter(X[y_pred1==2,0],X[y_pred1==2,1],s=100,c='green')
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100,c='yellow')
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

y_pred = ac.fit_predict(X)
print(y_pred)

plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c='red')
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='blue')
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='green')
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100,c='yellow')
plt.show()

###dendogram

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()