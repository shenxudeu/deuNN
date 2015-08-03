"""
Example of train a softmax classifier on MNIST dataset.
"""


import numpy as np
import sys

# easy install package
sys.path.append('../../deuNet/')

from deuNet.utils import np_utils
from deuNet.datasets import mnist
from deuNet.models import NN
from deuNet.layers.core import AffineLayer

import pdb

batch_size = 500
nb_classes = 10
nb_epoch = 20
learning_rate = 0.13

(train_X, train_y), (test_X, test_y) = mnist.load_data()
valid_X, valid_y = test_X, test_y

train_X = train_X.reshape((train_X.shape[0],-1))
valid_X = valid_X.reshape((valid_X.shape[0],-1))
test_X = test_X.reshape((test_X.shape[0],-1))

# make sure all data_X are in float32 for GPU use
train_X = train_X.astype('float32')
valid_X = valid_X.astype('float32')
test_X = test_X.astype('float32')
D = train_X.shape[1]
train_X /= 255
valid_X /= 255
test_X /= 255

# convert data_y to one-hot
train_y = np_utils.one_hot(train_y,nb_classes)
valid_y = np_utils.one_hot(valid_y,nb_classes)
test_y = np_utils.one_hot(test_y,nb_classes)

# NN architecture
model = NN()
model.add(AffineLayer(D, nb_classes, activation='softmax',reg_W=0.0001))

# Compile NN
print 'Compile NN ...'
model.compile(optimizer='SGD', loss='categorical_crossentropy',
        reg_type='L2', learning_rate=learning_rate)

# Train NN
model.fit(train_X, train_y, valid_X, valid_y,
        batch_size=batch_size, nb_epoch=nb_epoch, verbose=False)
