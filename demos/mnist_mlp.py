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
from deuNN.layers.core import AffineLayer,Dropout

import pdb

batch_size = 50
nb_classes = 10
nb_epoch = 10
learning_rate = 0.05
momentum = 0.9
lr_decay = 0.01
nesterov = False
reg_W = 0.001
nb_hidden1 = 500
nb_hidden2 = 500

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
model.add(AffineLayer(D, nb_hidden1, activation='sigmoid',reg_W=reg_W))
model.add(Dropout(0.2,nb_hidden1, uncertainty=True))
model.add(AffineLayer(nb_hidden1, nb_hidden2, activation='sigmoid',reg_W=reg_W))
model.add(Dropout(0.2,nb_hidden2, uncertainty=True))
model.add(AffineLayer(nb_hidden2, nb_classes, activation='softmax',reg_W=reg_W))

# Compile NN
print 'Compile NN ...'
model.compile(optimizer='SGD', loss='categorical_crossentropy',
        reg_type='L2', learning_rate=learning_rate,momentum=momentum,
        lr_decay=lr_decay,nesterov=nesterov)

# Train NN
model.fit(train_X, train_y, valid_X, valid_y,
        batch_size=batch_size, nb_epoch=nb_epoch, verbose=True)

# Test NN
model.get_test_accuracy(test_X, test_y)
#model.predict_uncertainty(test_X, 50)

# Save NN
model.save_model('mnist_mlp.h5')
