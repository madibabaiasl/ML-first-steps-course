#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 10 15:45:21 2022

@author: Mecharithm
"""

import numpy as np
import pandas as pd
from dog_or_car import dataset_collect
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras import regularizers

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

def model(input_shape):
    """
    Implements the forward propagation for the model
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    input_frame = tf.keras.Input(shape=input_shape)
    
    # Dense layers
    F1 = tfl.Dense(500, activation="relu", kernel_regularizer=regularizers.l2(0.03))(input_frame)
    
    ## 1 neuron in output layer. Hint: one of the arguments should be "activation='sigmoid'" 
    output_layer = tfl.Dense(2, activation="softmax")(F1)
    
    model = tf.keras.Model(inputs=input_frame, outputs=output_layer)
    return model
  

"""

 Extract dataset using dataset_collect function & feature scaling
 
"""

# Load the dataset
train_X, train_y, test_X, test_y = dataset_collect()

# Flatten input features
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

# Scale your dataset (to obtain better results)
train_X = train_X / 255.
test_X = test_X / 255.

# One hot representation
train_y = convert_to_one_hot(train_y)
test_y = convert_to_one_hot(test_y)

"""

 Construct NN model for image classification
 Train the model
 Display training and testing accuracies
 
"""

# Model construction
nn_model = model(train_X.shape[1:])
nn_model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.05),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
nn_model.summary()

# Model training
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(train_X.shape[0])
test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y)).batch(test_X.shape[0])
history = nn_model.fit(train_dataset, epochs=1000, validation_data=test_dataset)

"""

 Display learning curves
 
"""

# Plot learning curves
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
