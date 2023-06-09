#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 04:18:23 2022

@author: samer
"""

from dog import dataset_collect
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    # Sigmoid equation
    s = 1 / (1 + np.exp(-z))
    
    return s

def init_weights(dim):
    # Initialize weights randomly
    w = np.zeros([dim, 1], dtype = float)
    b = 0.0

    return w, b

def fwd_bwd(w, b, X, Y):
    # Number of examples
    m = X.shape[1]
    
    # FORWARD PROPAGATION (From X TO Cost)
    A = sigmoid(w.T @ X + b)
    J = (-1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

    # BACKWARD PROPAGATION (To find gradient)
    dw = (1/m) * (X@(A-Y).T)
    db = (1/m) * np.sum(A-Y)
    J = np.squeeze(np.array(J))

    
    grads = {"dw": dw,
             "db": db}
    
    return grads, J

def min_obj(w, b, X, Y, max_iter=100, alpha=0.009):
    costs = []
    
    # Loop for num_iterations times
    for i in range(max_iter):
        grads, J = fwd_bwd(w, b, X, Y)
        
        # Get derivatives w.r.t weight
        dw = grads["dw"]
        db = grads["db"]
        
        # Update weights
        w = w - alpha * dw
        b = b - alpha * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(J)
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    hypo = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute activations "probability of a dog image"
    A = sigmoid(w.T@X + b)
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to true/false predictions p[0,i]
        if A[0, i] > 0.5:
            hypo[0, i] = 1
        else:
            hypo[0, i] = 0       
    
    return hypo

# Load the dataset
train_X, train_y, test_X, test_y = dataset_collect()

# Flatten input features
train_X = train_X.reshape(train_X.shape[0], -1).T
test_X = test_X.reshape(test_X.shape[0], -1).T

# Scale your dataset (to obtain better results)
train_X = train_X / 255.
test_X = test_X / 255.

# Learning
max_iter = 2000
w, b = init_weights(train_X.shape[0]) #--> Initialize weights
params, grads, costs = min_obj(w, b, train_X, train_y, max_iter, 0.005)
w = params["w"]
b = params["b"]

# Predict train/test examples given and print accuracies
pred_train = predict(w, b, train_X)
pred_test = predict(w, b, test_X)
train_acc = (train_y.shape[1] - np.count_nonzero(pred_train - train_y)) / train_y.shape[1]
test_acc = (test_y.shape[1] - np.count_nonzero(pred_test - test_y)) / test_y.shape[1]
print('training accuracy: ', train_acc)
print('testing accuracy: ', test_acc)

# Print costs
plt.plot(np.arange(0, max_iter, 100), costs)
plt.xlabel('iterations')
plt.ylabel('cost')
plt.show()
