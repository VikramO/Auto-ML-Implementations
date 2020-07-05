# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 2020
Updated on Tue Jun 30 2020

Python file to handle training, testing, and utilization of ML models.
Eventually, cloud computation resources will be used, but local resources
will be used for current development phase.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.util import nest
import cython
import pandas as pd
import numpy as np

from autokeras import blocks
from autokeras import graph as graph_module
from autokeras import nodes as input_module
from autokeras import tuners
from autokeras.engine import head as head_module
from autokeras.engine import node as node_module
from autokeras.engine import preprocessor
from autokeras.engine import tuner
from autokeras.nodes import Input
from autokeras.utils import data_utils

TUNER_CLASSES = {
    'bayesian': tuners.BayesianOptimization,
    'random': tuners.RandomSearch,
    'hyperband': tuners.Hyperband,
    'greedy': tuners.Greedy,
}

def get_tuner_class(tuner):
    if isinstance(tuner, str) and tuner in TUNER_CLASSES:
        return TUNER_CLASSES.get(tuner)
    else:
        raise ValueError('The value {tuner} passed for argument tuner is invalid, '
                         'expected one of "greedy", "random", "hyperband", '
                         '"bayesian".'.format(tuner=tuner))

#write generalized code for dataset generation above
#assuming dataset is already defined as 'df'

"""
Firstly get some data.
"""

#use MNIST dataset for CNN class to see if this code works
num_classes = 10
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#simple normalization
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

#reshape data for compatibility with CNN
x_train=np.reshape(x_train, (60000,28,28,1))
x_test=np.reshape(x_test, (10000,28,28,1))

#one-hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
We can generalize the data preprocessing and put it in its own class later.
Now build a baseline model. We can create an algorithm which creates all sorts of architectures later.
"""

#build a crappy baseline model which (attempts to) classify the MNIST images
def build_cnn(self):
    input_data = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(16, kernel_size=(2,2), strides=(1,1), data_format='channels_last', activation='relu')(input_data)
    x = layers.GlobalMaxPooling2D(data_format='channels_last')(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=input_data, outputs=outputs, name="Baseline")
    return model

def build_rnn(self):
    #build some RNN
    return model

"""
Replace the above functions with a proprietary algorithms which build optimal neural networks.
"""

class CNN():
    def __init__(self):
        nn = build_cnn(self)
        self.compile = nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.train = nn.fit(x_train, y_train, epochs=1, batch_size=10000, validation_data=(x_test, y_test))
        self.test = nn.evaluate(x_test, y_test)
    

class RNN():
    def __init__(self):
        nn = build_rnn(self)
        self.compile = nn.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.train = nn.fit(x_train, y_train, epochs=1, batch_size=5, validation_data=(x_test, y_test))
        self.test = nn.evaluate(x_test, y_test)
