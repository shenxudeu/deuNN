"""
Example of training a ConvNet with Batch-Norm (BN) on MNIST dataset
"""

import numpy as np
import sys

sys.path.append('../../deuNet/')

from deuNet.utils import np_utils
from deuNet.datasets import mnist
from deuNet.models import NN
from deuNet.layers.core import AffineLayer, Dropout
from deuNet.layers.convolutional import Convolution2D,Flatten,MaxPooling2D
from deuNet.layers.batch_normalization import BatchNormalization

import pdb
np.random.seed(1337)

MODE = 1 # MODE -> 0: normal conv, 1: conv + BN

batch_size = 32
nb_classes = 10
nb_epoch = 20
if MODE == 0:
    learning_rate = 0.01
else:
    learning_rate = 1.00
momentum = 0.
lr_decay = 0.
nesterov = False
rho = 0.
reg_W = 0.

checkpoint_fn = '.trained_convnet_bn.h5'

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
if MODE == 0:
    model.add(Convolution2D(32,1,4,4, border_mode='valid',
        init='glorot_uniform',activation='relu', reg_W=0)) # output: (28-4+1)x(28-4+1) = 25x25x32
    model.add(MaxPooling2D(pool_size=(2,2))) # output: 13x13x32
    
    model.add(Convolution2D(64,32,5,5, border_mode='valid',
        init='glorot_uniform',activation='relu', reg_W=0)) # output: (13-5+1)x(13-5+1) = 9x9x64
    model.add(MaxPooling2D(pool_size=(2,2))) # output: 5x5*64

    model.add(Flatten())
    model.add(AffineLayer(64*5*5,200,init='glorot_uniform',activation='relu',reg_W=0))
    model.add(AffineLayer(200,10,init='glorot_uniform',activation='softmax',reg_W=0))

if MODE == 1:
    model.add(Convolution2D(32,1,4,4, border_mode='valid',
        init='glorot_uniform',activation='linear', reg_W=0)) # output: (28-4+1)x(28-4+1) = 25x25x32
    model.add(BatchNormalization((1,32,1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) # output: 13x13x32
    
    model.add(Convolution2D(64,32,5,5, border_mode='valid',
        init='glorot_uniform',activation='linear', reg_W=0)) # output: (13-5+1)x(13-5+1) = 9x9x64
    model.add(BatchNormalization((1,64,1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) # output: 5x5*64

    model.add(Flatten())
    model.add(AffineLayer(64*5*5,200,init='glorot_uniform',activation='linear',reg_W=0))
    model.add(BatchNormalization((1,200),activation='relu'))
    model.add(AffineLayer(200,10,init='glorot_uniform',activation='softmax',reg_W=0))


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


