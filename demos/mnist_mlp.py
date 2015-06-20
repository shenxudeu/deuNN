"""
Example of train a 2-layers Neural Network classifier on MNIST dataset.
"""


import numpy as np
import sys

# easy install package
sys.path.append('../../deuNN/')

from deuNN.utils import np_utils
from deuNN.datasets import mnist
from deuNN.models import NN
from deuNN.layers.core import AffineLayer

import pdb

batch_size = 50
nb_classes = 10
nb_epoch = 20
learning_rate = 0.01
nb_hidden = 500

[train_set, valid_set, test_set] = mnist.load_data()
[train_X, train_y] = train_set
[valid_X, valid_y] = valid_set
[test_X, test_y] = test_set

# make sure all data_X are in float32 for GPU use
train_X = train_X.astype('float32')
valid_X = valid_X.astype('float32')
test_X = test_X.astype('float32')
D = train_X.shape[1]

# convert data_y to one-hot
train_y = np_utils.one_hot(train_y,nb_classes)
valid_y = np_utils.one_hot(valid_y,nb_classes)
test_y = np_utils.one_hot(test_y,nb_classes)

# NN architecture
model = NN()
model.add(AffineLayer(D, nb_hidden, activation='sigmoid',reg_W=0.0001))
model.add(AffineLayer(nb_hidden, nb_classes, activation='softmax',reg_W=0.0001))

# Compile NN
print 'Compile NN ...'
model.compile(optimizer='SGD', loss='categorical_crossentropy',
        reg_type='L2', learning_rate=learning_rate)

# Train NN
model.fit(train_X, train_y, valid_X, valid_y,
        batch_size=batch_size, nb_epoch=nb_epoch, verbose=False)

# Save NN
model.save_model('mnist_mlp.h5')
