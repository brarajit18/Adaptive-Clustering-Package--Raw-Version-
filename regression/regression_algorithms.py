# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:07:51 2020

@author: ajitp
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

X = X[:100, [1,3]]
y = y[:100]

from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression

m1 = Perceptron(random_state=10)
m2 = LogisticRegression(random_state=10, solver='sag', penalty="none",max_iter=1000)
m3 = SGDClassifier(loss="log", penalty="none",tol=0.0001,average=10)

m1.fit(X,y)
m2.fit(X,y)
m3.fit(X,y)

m1.score(X,y)
m2.score(X,y)
m3.score(X,y)

w0 = m1.intercept_[0]
w1 = m1.coef_[0,0]
w2 = m1.coef_[0,1]

b0 = m2.intercept_[0]
b1 = m2.coef_[0,0]
b2 = m2.coef_[0,1]

c0 = m3.intercept_[0]
c1 = m3.coef_[0,0]
c2 = m3.coef_[0,1]

x1 = np.array([X[:,0].min(),X[:,0].max()])

x2_p = (-w1/w2)*x1 - (w0/w2)
x2_l = (-b1/b2)*x1 - (b0/b2)
x2_s = (-c1/c2)*x1 - (c0/c2)


plt.scatter(X[y == 0, 0], X[y == 0, 1], s = 100, c = 'red', label = '+')
plt.scatter(X[y == 1, 0], X[y == 1, 1], s = 100, c = 'blue', label = '-')
plt.plot(x1,x2_p,c='black', linewidth=4)
plt.plot(x1,x2_l,c='green', linewidth=4)
plt.plot(x1,x2_s,c='blue', linewidth=4)
plt.legend()