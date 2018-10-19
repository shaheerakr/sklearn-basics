# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:57:45 2018

@author: Shaheer Akram
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
iris=load_iris()
X=iris.data
y= iris.target
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
knn=KNeighborsClassifier(n_neighbors=11,n_jobs=-1)
knn.fit(X_train,y_train)
y_pred= knn.predict(X_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))
import matplotlib.pyplot as pl
pl.plot(y_pred,y_test)
from sklearn.linear_model import LogisticRegression
lreg= LogisticRegression()
lreg.fit(X_train,y_train)
y_pred=lreg.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))
X_train.shape
score=[]
k_range=range(1,30)
for i in k_range:
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    score.append(metrics.accuracy_score(y_test,y_pred))
import matplotlib.pyplot as plt
plt.plot(k_range,score)
from sklearn.model_selection import GridSearchCV
k_range=range(1,25)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn,param_grid,cv=10,n_jobs=-1,scoring='accuracy',return_train_score=False)
grid.fit(X,y)
import pandas as pd
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
grid_mean_scores = grid.cv_results_['mean_test_score']
pl.plot(k_range,grid_mean_scores)
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)
k_range=range(1,30)
weight_options = ['uniform', 'distance']
param_grid = dict(n_neighbors=k_range, weights=weight_options)
print(param_grid)
grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy',n_jobs=-1,return_train_score=False)
grid.fit(X,y)
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)
## to save time we use randomize search cv
from sklearn.model_selection import RandomizedSearchCV
param_dist = dict(n_neighbors=k_range, weights=weight_options)
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5, return_train_score=False)
rand.fit(X, y)
pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(rand.best_score_)
print(rand.best_params_)
best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, return_train_score=False)
    rand.fit(X, y)
    best_scores.append(round(rand.best_score_, 3))
print(best_scores)