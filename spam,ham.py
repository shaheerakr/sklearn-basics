# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 00:12:12 2018

@author: Shaheer Akram
"""

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

data=pd.read_csv('SMSSpamCollection',sep='\t',names=['status','message'])
