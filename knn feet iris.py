# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:37:08 2018

@author: Shaheer Akram
"""

from sklearn.datasets import load_iris
iris=load_iris()
type(iris)
print (iris.data)
X=iris.data
y=iris.target
print (X.shape)
print (y.shape)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
print(knn)
knn.fit(X,y)
a=[3,5,4,2]
print(knn.predict([[5.9,3.0,5.1,1.8]]))
print(iris.target)