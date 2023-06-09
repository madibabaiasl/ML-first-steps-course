#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 03:23:24 2022

@author: samer
"""
# Import the necessary libraries
import os
from PIL import Image
from numpy import asarray, empty, append, array

def dataset_collect():
    # Initialize training and testing datasets
    train_X = empty((0, 60, 60, 3), int)
    train_y = array([])
    test_X = empty((0, 60, 60, 3), int)
    test_y = array([])
    
    # get the path/directory for training dataset & load it
    folder_dir = os.getcwd() + '/train'
    for images in os.listdir(folder_dir):
        # check if the image ends with jpg
        if (images.endswith(".jpg")):
            img = Image.open(os.getcwd() + '/train/' + images) #--> Open image
            img = img.resize((60, 60)) #--> Resize image to a reasonable size
            train_X = append(train_X, [asarray(img)], axis = 0) #--> Convert to numpy array
            if (images.startswith("dogs")):
                train_y = append(train_y, 1)
            else:
                train_y = append(train_y, 0)
                
    # get the path/directory for testing dataset & load it
    folder_dir = os.getcwd() + '/test'
    for images in os.listdir(folder_dir):
        # check if the image ends with jpg
        if (images.endswith(".jpg")):
            img = Image.open(os.getcwd() + '/test/' + images) #--> Open image
            img = img.resize((60, 60)) #--> Resize image to a reasonable size
            test_X = append(test_X, [asarray(img)], axis = 0) #--> Convert to numpy array
            if (images.startswith("dogs")):
                test_y = append(test_y, 1)
            else:
                test_y = append(test_y, 0)
                
    # Reshape labels
    train_y = train_y.reshape(train_y.shape[0], 1).T
    test_y = test_y.reshape(test_y.shape[0], 1).T
                
    return train_X, train_y, test_X, test_y
