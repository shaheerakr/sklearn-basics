# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:03:42 2020

@author: Shaheer Akram
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = load_breast_cancer()

data = pd.DataFrame(data = cancer['data'],columns = cancer['feature_names'])

scaler = StandardScaler()

scaler.fit(data)

scaled_data = scaler.transform(data)

#PCA
pca = PCA(n_components = 2)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)


#ploting
plt.figure(figsize = (15,10))
plt.scatter(pca_data[:,0],pca_data[:,1],c=cancer["target"])
plt.xlabel("First Principal Componet")
plt.ylabel("Second Principal Component")


arr =  pca.components_
df_comp = pd.DataFrame(pca.components_,columns = cancer['feature_names'])

plt.figure(figsize = (15,10))
sns.heatmap(df_comp,cmap = 'plasma')
