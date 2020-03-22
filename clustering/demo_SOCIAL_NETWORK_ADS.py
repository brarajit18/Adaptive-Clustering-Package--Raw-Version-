# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from kMeans import kMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../datasets/Social_Network_Ads.csv')

df = df[['Age', 'EstimatedSalary']]

#sns.scatterplot(x=df1['Age'], y=df1['EstimatedSalary'])

kn = kMeans(df, 3, n_iter=100)
clust = kn.apply_clustering()
centroids = kn.get_centroids()
labels = kn.get_labels()

#labs = np.array(labs)
#labs[labs==0]

sns.scatterplot(x=clust[labels==0, 0], y=clust[labels==0, 1], color = 'blue', label='C0')
sns.scatterplot(x=clust[labels==1, 0], y=clust[labels==1, 1], color = 'red', label='C1')
sns.scatterplot(x=clust[labels==2, 0], y=clust[labels==2, 1], color = 'green', label='C2') 
sns.scatterplot(x=centroids[:,0], y=centroids[:,1], color = 'black', s=200, label='Centroids')
plt.show()