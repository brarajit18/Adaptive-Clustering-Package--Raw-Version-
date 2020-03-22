# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from kMedoids import kMedoids
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../datasets/Social_Network_Ads.csv')

df1 = df[['Age', 'EstimatedSalary']].copy()

#sns.scatterplot(x=df1['Age'], y=df1['EstimatedSalary'])

kn = kMedoids(df1, 3)
kn.select_centroids()
kn.make_clusters()
clust = kn.get_clusters()
centroids = kn.get_centroids()


sns.scatterplot(x=clust.loc[clust.labels==0, 'Age'], y=clust.loc[clust.labels==0, 'EstimatedSalary'], color = 'blue', label='C0')
sns.scatterplot(x=clust.loc[clust.labels==1, 'Age'], y=clust.loc[clust.labels==1, 'EstimatedSalary'], color = 'red', label='C1')
sns.scatterplot(x=clust.loc[clust.labels==2, 'Age'], y=clust.loc[clust.labels==2, 'EstimatedSalary'], color = 'green', label='C2') 
sns.scatterplot(x=centroids[:,0], y=centroids[:,1], color = 'black', s=200, label='Centroids')
plt.show()
