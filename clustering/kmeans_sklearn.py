# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:35:26 2020

@author: ajitp
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('../datasets/Social_Network_Ads.csv')

df1 = df[['Age', 'EstimatedSalary']].copy()

model = KMeans(n_clusters = 3)
model.fit(df1)
model.cluster_centers_
model.inertia_

labels = model.predict(df1)

sns.scatterplot(x=df1.iloc[labels==0, 0], y=df1.iloc[labels==0, 1], color = 'blue', label='C0')
sns.scatterplot(x=df1.iloc[labels==1, 0], y=df1.iloc[labels==1, 1], color = 'red', label='C1')
sns.scatterplot(x=df1.iloc[labels==2, 0], y=df1.iloc[labels==2, 1], color = 'green', label='C2') 
sns.scatterplot(x=model.cluster_centers_[:,0], y=model.cluster_centers_[:,1], color = 'black', s=200, label='Centroids')
plt.show()