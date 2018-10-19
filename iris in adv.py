# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:38:56 2018

@author: Shaheer Akram
"""

from sklearn.linear_model import LogisticRegression
lReg = LogisticRegression()
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y =  iris.target
lReg.fit(X,y)
new= [[3,5,4,2],[5,4,3,2]]
lReg.predict(new)