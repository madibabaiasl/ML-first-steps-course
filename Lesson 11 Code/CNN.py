#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 13 15:45:21 2022

@author: mecharithm
"""

import numpy as np
import pandas as pd
from math import floor
import tensorflow as tf
import tensorflow.keras.layers as tfl
#from tensorflow.keras import regularizers

def convert_to_one_hot(y):
    """
    converts y into one hot reprsentation.

    Parameters
    ----------
    y : numpy array
        An array containing integer values.

    Returns
    -------
    one_hot : numpy.ndarray
        A numpy.ndarray object, which is one-hot representation of y.

    """
    max_value = max(y)
    min_value = min(y)
    length = len(y)
    one_hot = np.zeros((length, (max_value - min_value + 1)))
    one_hot[np.arange(length), y - min_value] = 1
    return one_hot 

def convert_from_one_hot(y):
    """
    converts y back from one hot/probability reprsentation to normal.

    Parameters
    ----------
    y : numpy array
        An array containing class probabilities/one hot form.

    Returns
    -------
    one_hot : numpy.ndarray
        A numpy.ndarray object, which is a normal representation of y.

    """
    y_origin = np.argmax(y, axis = 1)
    return y_origin

def process(Ct_list, trans_sz, frame_sz):
    """

    processes raw dataset to produce a new dataset with no transitional phase or
    incomplete widnows.

    Parameters
    ----------
    Ct_list : list
        A list containing names of dataset csv files.
    trans_sz : int
        Transitional phase window size (multiples of 0.002 sec.)
    frame_sz : int
        Size of input frame to CNN

    Returns
    -------
    X : numpy.ndarray
        A numpy.ndarray object, which is the processed features vector
    y : numpy.ndarray
        A numpy.ndarray object, which is the one hot representation of labels
     
    """
    # Construct dataset numpy arrays
    for i in range(len(Ct_list)):
        # Import circuit data
        ct = pd.read_csv('gait mode/' + Ct_list[i] 
                         + '.csv').to_numpy() #--> Current circuit
        ct = ct[:, [0, 2, 3, 6, 8, 9, 12, 14, 15, 18, 20, 21, 25, 26, 29, 44, 45, 
                    46, 47, 48]] #--> Usable features from IMU and Goniometer + label
        
        # Remove transitional phase datapoints to avoid confusion (~ 1 sec.)
        I = np.argwhere(np.append(0, np.diff(ct[:, -1])) != 0).T[0] 
            #--> Search for indeces of mode transition
        for j in range(0, len(I)):
            ct = np.delete(ct, np.s_[I[j] - trans_sz : I[j] + trans_sz], 0)
                #--> Delete transitional period from dataset
            I = I - trans_sz * 2 #--> Update index values after transition deletion
        
        # Remove incomplete frames from each walking phase
        I = np.append(np.argwhere(np.append(0, np.diff(ct[:, -1])) != 0).T[0],
                      ct.shape[0] - 1) #--> Index of current mode final reading
        for j in range(0, len(I)):
            size_before = ct.shape[0] #--> Circuit size before cutting off residuals
            ct = np.delete(ct, np.s_[floor(I[j]/frame_sz) * frame_sz : I[j]], 0)
                #--> Delete residuals/incomplete frames 
            size_after = ct.shape[0] #--> Circuit size after deleting residuals
            I = I - (size_before - size_after) #--> Update index values after deletion
        ct = np.delete(ct, ct.shape[0] - 1, 0) #--> Remove last residual
        
        # Append current circuit to dataset
        if i == 0:
            Dataset_numpy = ct #--> Initialize training array using first circuit
        else:
            Dataset_numpy = np.append(Dataset_numpy, ct, axis = 0) #--> Append other cts
        
        # Extract training features and labels from dataset
        m = int(Dataset_numpy.shape[0] / frame_sz) #--> Number of training examples
        nsens = Dataset_numpy.shape[1] - 1 #--> Number sensor readings
        nc = 1 #--> Number of channels (only one channel)
        X = Dataset_numpy[:, 0:-1].reshape(m, frame_sz, nsens, nc) #--> Features
        X = X.transpose(0, 2, 1, 3) 
            #--> Transpose to get (no. of exp, no. of sensors, no. of samples, no. of channels)
        y = np.int64(Dataset_numpy[0:Dataset_numpy.shape[0]:frame_sz, -1]) #--> Labels
        
        # Shuffle
        idx = np.arange(m) #--> Range of array indeces
        np.random.shuffle(idx) #--> Shuffle indeces
        X = X[idx, :, :] #--> Shuffled features
        y = convert_to_one_hot(y[idx]) #--> Shuffles labels
    
    # Return dataset
    return X, y

def model(input_shape):
    """
    Implements the forward propagation for the model
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    input_frame = tf.keras.Input(shape=input_shape)
    
    ## Convolutional layer (10 filters, 3x3)
    Z1 = tfl.Conv2D(10, 3, activation='tanh')(input_frame)
    
    ## Maxpooling layer (size = 3, stride = 2)
    max1 = tfl.MaxPooling2D(pool_size=3, strides=2, padding='valid')(Z1)
    
    ## FLATTEN
    F = tfl.Flatten()(max1)
    
    # Fully connected layers
    F1 = tfl.Dense(400, activation="relu")(F)
    D1 = tfl.Dropout(0.3)(F1)
    F2 = tfl.Dense(50, activation="relu")(D1)
    D2 = tfl.Dropout(0.3)(F2)
    
    # Softmax
    output_layer = tfl.Dense(7, activation="softmax")(D2)
    
    model = tf.keras.Model(inputs=input_frame, outputs=output_layer)
    return model
  
"""

 Extract dataset frames from csv files using pandas
 
 Remove unwanted emg features and keep IMU/Goniometer features
 
 Remove overlapped transition phases
 
 Remove incomplete input 1D frames
 
"""

# Determine training "seen" and testing "unseen" subjects
Train_list = ['AB194_Circuit_001_raw', 'AB194_Circuit_002_raw',
              'AB194_Circuit_003_raw', 'AB194_Circuit_004_raw',
              'AB194_Circuit_005_raw', 'AB194_Circuit_006_raw',
              'AB194_Circuit_007_raw', 'AB194_Circuit_008_raw'] 

Test_list = ['AB194_Circuit_009_raw', 'AB194_Circuit_010_raw'] 

# Settings for dataset preprocessing
trans_sz = 500 #--> Transitional window size (~ 1 sec. from each activity)
frame_sz = 25 #--> Input frame length (~ 0.05 sec.)

# Construct dataset numpy arrays
# Training dataset with a 20% validation split
X_train, y_train = process(Train_list, trans_sz, frame_sz)

# Testing dataset
X_test, y_test = process(Test_list, trans_sz, frame_sz)

"""

 Construct CNN model for gait mode detection
 Train the model
 Display training, validation and testing accuracies
 
"""

# CNN model construction
cnn2d_model = model(X_train.shape[1:])
cnn2d_model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.05),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
cnn2d_model.summary()

# Model training, resulting losses and accuracies (training, validation and testing)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(X_train.shape[0])
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(X_test.shape[0])
history = cnn2d_model.fit(train_dataset, epochs=10000, validation_data=test_dataset)

"""

 Display learning curves
 Display confusion matrices for training, validation and testing
 
"""

# Plot learning curves
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
