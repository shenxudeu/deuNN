"""
Example of train a 2-layers Neural Network classifier on MNIST dataset.
"""


import numpy as np
import sys

# easy install package
sys.path.append('../../deuNet/')

from deuNet.utils import np_utils
from deuNet.datasets import mnist
from deuNet.models import NN
from deuNet.layers.core import AffineLayer,Dropout

import pdb

batch_size = 50
nb_classes = 10
nb_epoch = 100
learning_rate = 0.05
momentum = 0.9
lr_decay = 0.01
nesterov = True
rho = 0.9
reg_W = 0.001
nb_hidden1 = 500
nb_hidden2 = 500

checkpoint_fn = '.trained_net.h5'
log_fn = '.trained_log.log'

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
model = NN(checkpoint_fn,log_fn)
model.add(AffineLayer(D, nb_hidden1, activation='sigmoid',reg_W=reg_W))
model.add(Dropout(0.2,nb_hidden1, uncertainty=True))
model.add(AffineLayer(nb_hidden1, nb_hidden2, activation='sigmoid',reg_W=reg_W))
model.add(Dropout(0.2,nb_hidden2, uncertainty=True))
model.add(AffineLayer(nb_hidden2, nb_classes, activation='softmax',reg_W=reg_W))

# Compile NN
print 'Compile NN ...'
model.compile(optimizer='RMSprop', loss='categorical_crossentropy',
        reg_type='L2', learning_rate=learning_rate,momentum=momentum,
        lr_decay=lr_decay,nesterov=nesterov, rho = rho)

# Train NN
model.fit(train_X, train_y, valid_X, valid_y,
        batch_size=batch_size, nb_epoch=nb_epoch, verbose=True)

# Test NN
model.get_test_accuracy(test_X, test_y)
#model.predict_uncertainty(test_X, 50)

# Save NN
#model.save_model('mnist_mlp.h5')
