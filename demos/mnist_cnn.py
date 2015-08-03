"""
Example of training a ConvNet Classifier on MNIST dataset.
"""

import numpy as np
import sys

sys.path.append('../../deuNet/')

from deuNet.utils import np_utils
from deuNet.datasets import mnist
from deuNet.models import NN
from deuNet.layers.core import AffineLayer, Dropout
from deuNet.layers.convolutional import Convolution2D,Flatten,MaxPooling2D

import pdb
np.random.seed(1337)

batch_size = 128
nb_classes = 10
nb_epoch = 50
learning_rate = 0.01
momentum = 0.
lr_decay = 0.
nesterov = False
rho = 0.
reg_W = 0.

checkpoint_fn = '.trained_convnet.h5'

#[train_set, valid_set, test_set] = mnist.load_data()
(train_X, train_y), (test_X, test_y) = mnist.load_data()
valid_X, valid_y = test_X, test_y

# make sure all data_X are in float32 for GPU use
train_X = train_X.astype('float32')
valid_X = valid_X.astype('float32')
test_X = test_X.astype('float32')
train_X /= 255
valid_X /= 255
test_X /= 255

# Reshape input to 4D volume
train_X = train_X.reshape((-1,1,28,28))
valid_X = valid_X.reshape((-1,1,28,28))
test_X = test_X.reshape((-1,1,28,28))

# convert data_y to one-hot
train_y = np_utils.one_hot(train_y,nb_classes)
valid_y = np_utils.one_hot(valid_y,nb_classes)
test_y = np_utils.one_hot(test_y,nb_classes)

# NN architecture
model = NN(checkpoint_fn)
model.add(Convolution2D(32,1,3,3, border_mode='full',
                        init='glorot_uniform',activation='relu', reg_W=0))
model.add(Convolution2D(32,32,3,3, border_mode='valid',
                        init='glorot_uniform',activation='relu', reg_W=0))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.2, uncertainty=False))

model.add(Flatten())
model.add(AffineLayer(32*196,128,init='glorot_uniform',activation='relu',reg_W=0))
model.add(AffineLayer(128,10,init='glorot_uniform',activation='softmax',reg_W=0))

# Compile NN
print 'Compile ConvNet ...'
model.compile(optimizer='SGD', loss='categorical_crossentropy',
        reg_type='L2', learning_rate=learning_rate, momentum=momentum,
        lr_decay=lr_decay, nesterov=nesterov, rho=rho)

# Train NN
model.fit(train_X, train_y, valid_X, valid_y,
        batch_size=batch_size, nb_epoch=nb_epoch, verbose=True)

# Test NN
model.get_test_accuracy(test_X, test_y)


