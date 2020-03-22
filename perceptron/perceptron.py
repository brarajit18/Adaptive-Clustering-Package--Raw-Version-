# -*- coding: utf-8 -*-

import array
import numpy as np
import pandas as pd

X = np.array([[1, 0, 0],
     [1, 0, 1],
     [1, 1, 0],
     [1, 1, 1]
    ])

y = [0, 0, 0, 1]



LR = 0.5
thresh_tolerance = 0.001

"""
# Example with one perceptron:

## LECTURE NOTES FOR REGRESSION:
Regularized regression is produced by ridge-regression (lasso)
"""
def generate_weights(X):
    W = np.ones(X.shape[1])
    return W

def compute_net(X_row, W):
    net = 0
    for i in range(len(X_row)):
        net += X_row[i] * W[i]
    return net

def compute_y_hat(net):
    if net >= 0:
        y_hat = 1
    else:
        y_hat = 0
    return y_hat


def update_weights(X_row, W, row_num, y_hat, y):
    for i in range(len(W)):
        W[i] += LR*X_row[i]*(y-y_hat)
    return W


def check_convergence(W_old, W, thresh_tolerance):
    p_sum = np.sum(W_old) - np.sum(W)
    if ( p_sum > -(thresh_tolerance) ) and ( p_sum < thresh_tolerance ):
        return True
    else:
        return False
    

def fit(X, y, epoc=10, verbose = False):
        W = generate_weights(X)
        for e in range(epoc):
            print ("Epoc: ", e)
            for row_num in range(len(X)):
                net = compute_net(X[row_num], W)
                W_old = W
                y_hat = compute_y_hat(net)
                W = update_weights(X[row_num], W, row_num, y_hat, y[row_num])
                if (verbose == True): 
                    output = [net, W_old, W, np.array([y_hat, y[row_num]])]
                    print (output, "\n")
            if (e >= epoc/2) and check_convergence(W_old, W, thresh_tolerance):
                print ("Reached the Convergence Level")
                return W
        return W

def predict(X_test, y_test):
    pass


    
model = fit(X, y, epoc=10, verbose=True)