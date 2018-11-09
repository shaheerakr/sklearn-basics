# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 22:25:08 2018

@author: Shaheer Akram
"""

import pandas as pd
data = pd.read_csv("out.xls",index_col = 0)
data.head()
data = data.replace("Positive",1)
data = data.replace("Negative",0)
data = data.replace("Neutral",2)
X = data["Sentence"]
y = data["Sentiment"]
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_score
cv = CountVectorizer(min_df=1)
cv = TfidfVectorizer(min_df=1)
X_traincv = cv.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_traincv,y,random_state=1,test_size=0.2)
lreg = linear_model.LogisticRegression()
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
svm = svm.SVC()
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
scores = cross_val_score(lreg, X_traincv, y, cv=100, scoring="accuracy")
print(np.max(scores))
print(np.mean(scores))
scores_mnb = cross_val_score(mnb, X_traincv, y, cv=500, scoring="accuracy")
print(np.mean(scores_mnb))
sdg = linear_model.SGDClassifier()
sdg.fit(X_train,y_train)
sdg = linear_model.SGDClassifier()
sdg.fit(X_train,y_train)
y_pred = sdg.predict(X_test)
scores = cross_val_score(sdg,X_traincv,y,cv=100,scoring="accuracy")