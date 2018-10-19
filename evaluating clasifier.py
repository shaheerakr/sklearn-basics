# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 21:34:38 2018

@author: Shaheer Akram
"""

import pandas as pd
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
data = pd.read_csv('diabetes.csv',header=None,skiprows=[0])
data.columns = col_names
data.head()
feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
X = data[feature_cols]
y = data.label
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)
from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()
lreg.fit(X_train,y_train)
y_pred_class = lreg.predict(X_test)
##clasification accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred_class))
##null accuracy
y_test.value_counts()
y_test.mean()
1-y_test.mean()
# calculate null accuracy (for binary classification problems coded as 0/1)
max(y_test.mean(), 1 - y_test.mean())
# calculate null accuracy (for multi-class classification problems)
y_test.value_counts().head(1) / len(y_test)
print('true: ',y_test.values[0:25])
print('pred: ',y_pred_class[0:25])