"""
Example of train a 2-layers Neural Network classifier on CIFAR-10 dataset
"""
import numpy as np
import sys

np.random.seed(1984)
sys.path.append("../../deuNN/")

from deuNN.utils import np_utils
from deuNN.datasets import cifar10
from deuNN.models import NN
from deuNN.layers.core import AffineLayer, Dropout

import pdb

batch_size = 100
nb_classes = 10
nb_epoch = 20
learning_rate = 1e-2
momentum = None
lr_decay = None
nesterov = False
rho = 0.9
reg_W = 0.
nb_hidden1 = 32*32

checkpoint_fn = '.trained_cifar10_mlp.h5'

#[train_X, train_y,valid_X, valid_y, test_X, test_y] = cifar_10.load_data()
(train_X, train_y), (test_X, test_y) = cifar10.load_data()
valid_X,valid_y = test_X, test_y

nb_channels, nb_w, nb_h = train_X.shape[1], train_X.shape[2], train_X.shape[3]
train_X = train_X.reshape(-1,nb_channels*nb_w*nb_h).astype('float32')
valid_X = valid_X.reshape(-1,nb_channels*nb_w*nb_h).astype('float32')
test_X = test_X.reshape(-1,nb_channels*nb_w*nb_h).astype('float32')
D = train_X.shape[1]
train_X /= 255
valid_X /= 255
test_X  /= 255
# convert data_y to one-hot
train_y = np_utils.one_hot(train_y, nb_classes)
valid_y = np_utils.one_hot(valid_y, nb_classes)
test_y = np_utils.one_hot(test_y, nb_classes)

# NN architecture
model = NN(checkpoint_fn)
model.add(AffineLayer(D, nb_hidden1, init='glorot_uniform',activation='relu', reg_W = reg_W))
model.add(AffineLayer(nb_hidden1, nb_classes, init='glorot_uniform',activation='softmax',reg_W=reg_W))

# Compile NN
print 'Compile NN ...'
model.compile(optimizer='RMSprop', loss='categorical_crossentropy',
        reg_type='L2', learning_rate = learning_rate, momentum=momentum,
        lr_decay=lr_decay, nesterov=nesterov, rho=rho)

# Train NN
model.fit(train_X, train_y, valid_X, valid_y,
        batch_size=batch_size, nb_epoch=nb_epoch, verbose=True)

# Test NN
model.get_test_accuracy(test_X, test_y)

