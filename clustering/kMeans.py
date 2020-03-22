    # -*- coding: utf-8 -*-
"""k-Means Algorithm

This module implements the ordinary k-means algorithm. k-Means aglroithms is a 
clustering algorithm, which is used to segment input data in multiple groups 
based upon inter-cluster likelihood. 

Usage Example:
    This module requires the input data, containing only the columns used for 
    clustering and should contain no extra column. Following is the example of
    this module:
        
        df = pd.read_csv('../datasets/old_faithful.csv')

        X = df[['Age', 'EstimatedSalary']] # Clustering Attribute Selection
                
        kn = kMeans(X, 2, scale_data_boolean = True) # for 2 clusters

        clust = kn.apply_clusters()

        centroids = kn.get_centroids()

        labels = kn.get_labels()


Input Attributes:
    
    data: This attribute is mandatory, and contains the data in DataFrame or
    Array.
    
    k: This attributes indicates the number of clusters.

    n_iter: (Default value : 10) This attribute control the iteration to divide 
    data into clusters
    
    scale_data_boolean: (Default value: True) This attribute control the data 
    scaling. Please select "True" to scale the data, and select "False" to pass
    data without scaling.

Class Methods:
    
    Call for Library: Use the following syntax to call the kMeans library:
        from kMeans import kMeans
    
    Create class object: To create the class object, use the following syntax:
        km = kMeans(df, 3)

    Apply clustering: Apply clustering after creating the class method using
    following method:
        clust = km.apply_clusters()

    Centroid Data: To get the centroid_data, use the following syntax:
        centroids = km.get_centroids()
    
    Centroid Data Variable Access: Alternatively, variable centroids_ can be used
    to display the centroid data:
        centroids = km.centroids_

    Cluster Labels Data: To get the cluster labels, use the following syntax:
        labels = km.get_labels()

    Distance Data: To get the Euclidean Distance vector, use the following:
        distance = km.get_distances()
            

Todo:
    * For this module, following libraries are used:
    * Pandas, Numpy, StandardScaler, operator
    * Note: These libraries are already imported with class definition.
"""

import operator
import numpy as np
from sklearn.preprocessing import StandardScaler

class kMeans:
    def __init__(self, data, k, n_iter =  10, scale_data_boolean = True):
        self.data = np.array(data)
        self.k = k
        self.n_iter = n_iter
        self.n = self.data.shape[1]
        self.scale_data_boolean = scale_data_boolean
        self.apply_scaling()
    
    def apply_scaling(self):
        if self.scale_data_boolean == True:
            scaler = StandardScaler()
            self.data = scaler.fit_transform(self.data)        
        
    def select_centroids(self):
        self.centroids_ = self.data[np.random.randint(0, len(self.data), self.k), :]
        
    def distance(self, centroid, row):
        dis = 0
        for i in range(len(centroid)):
            dis += (centroid[i] - row[i]) ** 2
        return dis
    
    def update_centroids(self):
        data = np.array(self.data)
        labels = np.array(self.labels)
        for i in range(len(self.centroids_)):
            df_temp = data[labels==i]
            for m in range(df_temp.shape[1]):
                self.centroids_[i,m] = float( np.mean(df_temp[:,m]) )
            
    def make_clusters(self):
        for i in range(self.n_iter):
            self.distances = []
            self.labels = []
            for row_num in range(len(self.data)): 
                row = self.data[row_num]
                d = []
                for centroid in self.centroids_:
                    d.append(self.distance(centroid, row))
                min_index, min_value = min(enumerate(d), key=operator.itemgetter(1))
                self.distances.append(min_value)
                self.labels.append(min_index)
            self.update_centroids()

    def apply_clustering(self):
        self.select_centroids()
        self.make_clusters()            
        return self.get_clusters()

    def get_distances(self):
        return np.array(self.distances)
            
    def get_labels(self):
        return np.array(self.labels)
    
    def get_clusters(self):
        return np.array(self.data)

    def get_centroids(self):
        return np.array(self.centroids_)
    