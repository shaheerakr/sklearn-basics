# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 03:43:31 2018

@author: Shaheer Akram
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cross_validation import cross_val_score
data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv",index_col=0)
data.head()
sns.pairplot(data,x_vars=["TV","radio","newspaper"],y_vars=["sales"],kind="reg")
feature_col=["TV","radio","newspaper"]
feature_col2=["TV","radio"]
X=data[feature_col2]
y=data[["sales"]]
lreg=LinearRegression()
scores=cross_val_score(lreg,X,y,cv=10,scoring="mean_squared_error")
pos_scores=-scores
pos_scores2=-scores
sq2=np.sqrt(np.mean(pos_scores2))
x_pos= np.arange(len(pos_scores))
plt.bar(x_pos-0.2,np.sqrt(pos_scores),width=0.4,label="news")
plt.bar(x_pos+0.2,np.sqrt(pos_scores2),width=0.4,label="nonews")
plt.legend()