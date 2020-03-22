# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import operator
import numpy as np

class kMedoids:
    def __init__(self, data, k, n_iter =  10):
        self.data = data
        self.k = k
        self.n_iter = n_iter
        self.n = self.data.shape[1]
    
    def select_centroids(self):
        self.centroids = np.array(self.data.sample(self.k))
    
    def get_centroids(self):
        return self.centroids
    
    def distance(self, centroid, row):
        dis = 0
        #print ("\nDistance Method: ")
        #print ("Row: ", row)
        #print ("Centroid: ", centroid)
        #print ("Distnace Method Ends;\n")
        for i in range(len(centroid)):
            dis += (centroid[i] - row[i]) ** 2
        return dis
    
    def update_centroids(self):
        for i in range(len(self.centroids)):
            df_temp = self.data.loc[self.data.labels==i]
            df_temp = df_temp.iloc[:,:self.n]
            for m in range(df_temp.shape[1]):
                self.centroids[i,m] = float( np.median(df_temp.iloc[:,m]) )
                
    def make_clusters(self):
        self.data['distances'] = 0
        self.data['labels'] = 0
        for i in range(self.n_iter):
            for row_num in range(len(self.data)): 
                row = list(self.data.iloc[row_num,:self.n])
                #print ("Row: ", row)
                d = []
                for centroid in self.centroids:
                    #print ("Centroid: ", centroid)
                    d.append(self.distance(centroid, row))
                min_index, min_value = min(enumerate(d), key=operator.itemgetter(1))
                self.data['distances'][row_num] = min_value
                self.data['labels'][row_num] = min_index
            self.update_centroids()
            
    def get_clusters(self):
        return self.data
            
    
    def apply_clustering(self):
        self.select_centroids()
        self.cluster()
