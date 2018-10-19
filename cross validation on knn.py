# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 17:35:52 2018

@author: Shaheer Akram
"""

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
from sklearn.datasets import load_iris
iris= load_iris()
X= iris.data
y= iris.target
scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")
print(scores)
print(np.mean(scores))
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, y, cv = 10, scoring = "accuracy")
    k_scores.append(np.mean(scores))
print(k_scores)
import matplotlib.pyplot as plt
plt.plot(k_range,k_scores)
plt.xlabel("value of k for knn")
plt.ylabel("cross validation accuracy")