# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 02:09:46 2018

@author: Shaheer Akram
"""

import pandas as pd
data = pd.read_csv("Advertising.csv",index_col=0)
data.head()
data.tail()
import seaborn as sns
sns.pairplot(data,x_vars=["TV","radio","newspaper"],y_vars="sales",size=7,aspect=0.7,kind="reg")
feature_col=["TV","radio","newspaper"]
X= data[feature_col]
X.head()
y= data[["sales"]]
y.head()
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=1)
from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
lreg.fit(X_train,y_train)
print(lreg.coef_)
print(lreg.intercept_)
zip(feature_col,lreg.coef_)
y_pred=lreg.predict(X_test)
from sklearn import metrics
import numpy as np
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))