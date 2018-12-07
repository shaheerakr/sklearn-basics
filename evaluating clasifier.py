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
#clasification accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred_class))
#null accuracy
y_test.value_counts()
y_test.mean()
1-y_test.mean()
# calculate null accuracy (for binary classification problems coded as 0/1)
max(y_test.mean(), 1 - y_test.mean())
# calculate null accuracy (for multi-class classification problems)
y_test.value_counts().head(1) / len(y_test)
print('true: ',y_test.values[0:25])
print('pred: ',y_pred_class[0:25])
print(metrics.confusion_matrix(y_test,y_pred_class))
#!pip install pillow
from PIL import Image
img = Image.open("09confusionmatrix1.png")
img.show()
"""
True Positives (TP): we correctly predicted that they do have diabetes
True Negatives (TN): we correctly predicted that they don't have diabetes
False Positives (FP): we incorrectly predicted that they do have diabetes (a "Type I error")
False Negatives (FN): we incorrectly predicted that they don't have diabetes (a "Type II error")
"""
confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
# clasification accuracy
print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, y_pred_class))
# clasificaion error
print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, y_pred_class))
# sensetivity
print(TP/float(TP+FN))
print(metrics.recall_score(y_test,y_pred_class))
# specifitvity
print(TN/float(TN+FP))
#false positive OR 1-SPECIVITY
print(FP/float(TN+FP))
#PRISICION
print(TP/float(TP+FP))
print(metrics.precision_score(y_test,y_pred_class))
y_pred_lreg = lreg.predict(X_test)
print(y_pred_lreg[:10])
y_pred_lreg_prob = lreg.predict_proba(X_test)
print(y_pred_lreg_prob[:10])
import matplotlib.pyplot as plt
# histogram of predicted probabilities
plt.hist(y_pred_lreg_prob[:,1], bins=8)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')
from sklearn.preprocessing import binarize
new_y_pred_class = binarize([y_pred_lreg_prob[:,1]],0.3)[0]
y_pred_lreg_prob[:10,1]
print(new_y_pred_class[:10])
print(confusion)
print(metrics.confusion_matrix(y_test,new_y_pred_class))
# sensitivity has increased (used to be 0.24)
print(46 / float(46 + 16))
# specificity has decreased (used to be 0.91)
print(80 / float(80 + 50))
print(metrics.accuracy_score(y_test,new_y_pred_class))