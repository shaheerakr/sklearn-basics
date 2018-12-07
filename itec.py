# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 19:55:12 2018

@author: Shaheer Akram
"""
train_pos=[]
train_neg=[]
import glob as glob
import csv
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np
path='*.txt'
files=glob.glob(path)
for name in files:
    with open(name, encoding="utf-8") as f:
        train_neg.append(f.read())
        pass
print(len(train_pos))
f=open('all.csv','a')
a=csv.writer(f,delimiter=',')
data=[['num','review','polarity'],]
a.writerows(data)
f.flush()
for i in range(0,len(train_neg)):
    with open('all.csv','a',encoding="utf-8") as f:
        a=csv.writer(f,delimiter=',')
        data=[i+12499,train_neg[i],'0']
        a.writerow(data)
f.close()
data=pd.read_csv('all.csv',index_col='num')
X=data['review']
y=data['polarity']
cv = CountVectorizer(min_df=1,stop_words='english')
X_traincv = cv.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_traincv,y,random_state=1,test_size=0.25)
lreg = linear_model.LogisticRegression()
lreg.fit(X_train,y_train)
y_pred = lreg.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_pred = mnb.predict(X_test)
svm = svm.SVC()
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
k_range=range(1,25)
score=[]
for i in k_range:
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    score.append(metrics.accuracy_score(y_test,y_pred))
scores = cross_val_score(lreg, X_traincv, y, cv=100, scoring="accuracy")
print(np.max(scores))
print(np.mean(scores))
scores_mnb = cross_val_score(mnb, X_traincv, y, cv=500, scoring="accuracy")
print(np.mean(scores_mnb))
sdg = linear_model.SGDClassifier()
sdg.fit(X_train,y_train)
y_pred = sdg.predict(X_test)
scores = cross_val_score(sdg,X_traincv,y,cv=100,scoring="accuracy")