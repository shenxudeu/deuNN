"""
Example of train a 2-layers Neural Network classifier on CIFAR-10 dataset
"""
import numpy as np
import sys

sys.path.append("../../deuNet/")

from deuNet.utils import np_utils
from deuNet.datasets import cifar10
from deuNet.models import NN
from deuNet.layers.core import AffineLayer, Dropout
from deuNet.layers.convolutional import Convolution2D,Flatten,MaxPooling2D
from deuNet.layers.batch_normalization import LRN2D, BatchNormalization

import pdb
np.random.seed(1984)

batch_size = 128
nb_classes = 10
nb_epoch = 100
learning_rate = .01
w_scale = 0.01
momentum = 0.9
lr_decay = 0.1
nesterov = True
rho = 0.9
reg_W = 0.

checkpoint_fn = '.trained_cifar10_alex.h5'
log_fn = '.cifar10_alex.log'

#[train_X, train_y,valid_X, valid_y, test_X, test_y] = cifar_10.load_data()
(train_X, train_y), (test_X, test_y) = cifar10.load_data()
valid_X,valid_y = test_X, test_y

# convert data_y to one-hot
train_y = np_utils.one_hot(train_y, nb_classes)
valid_y = np_utils.one_hot(valid_y, nb_classes)
test_y = np_utils.one_hot(test_y, nb_classes)

train_X = train_X.astype("float32")
valid_X = valid_X.astype("float32")
test_X = test_X.astype("float32")
train_X /= 255
valid_X /= 255
test_X  /= 255

# size calculation
valid = lambda x, y, kernel, stride: ((x-kernel)/stride+1, (y-kernel)/stride+1)
full  = lambda x, y, kernel, stride: ((x+kernel)/stride-1, (y+kernel)/stride-1)
pool  = lambda x, y, kernel, stride: ((x-kernel)/stride+1, (y-kernel)/stride+1)

# NN architecture
model = NN(checkpoint_fn, log_fn)

model.add(Convolution2D(32,3,5,5, border_mode='same',subsample=(1,1),
    init='glorot_uniform',activation='relu', reg_W=reg_W))  
#model.add(BatchNormalization((1,32,1,1),activation='relu'))
nh, nw = (32, 32)
model.add(MaxPooling2D(pool_size=(3,3),stride=(2,2),ignore_border=True))
nh, nw = pool(nh, nw,3,2)
model.add(LRN2D(n=3))

model.add(Convolution2D(32,32,5,5, border_mode='same',subsample=(1,1),
    init='glorot_uniform',activation='relu', reg_W=reg_W))  
#model.add(BatchNormalization((1,32,1,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),stride=(2,2),ignore_border=True))
nh, nw = pool(nh, nw,3,2)
model.add(LRN2D(n=3))

model.add(Convolution2D(64,32,5,5, border_mode='same',subsample=(1,1),
    init='glorot_uniform',activation='relu', reg_W=reg_W))  
#model.add(BatchNormalization((1,64,1,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),stride=(2,2),ignore_border=True))
nh, nw = pool(nh, nw,3,2)
model.add(LRN2D(n=3))

model.add(Flatten())
#model.add(AffineLayer(64*nh*nw, 10,activation='relu',reg_W=reg_W, init='glorot_uniform'))
#model.add(Dropout(0.5, uncertainty=False))
model.add(AffineLayer(64*nh*nw, nb_classes,activation='softmax',reg_W=reg_W,init='glorot_uniform'))


# Compile NN
print 'Compile NN ...'
model.compile(optimizer='SGD', loss='categorical_crossentropy',
        reg_type='L2', learning_rate = learning_rate, momentum=momentum,
        lr_decay=lr_decay, nesterov=nesterov, rho=rho)

# Train NN
model.fit(train_X, train_y, valid_X, valid_y,
        batch_size=batch_size, nb_epoch=nb_epoch, verbose=True)

# Test NN
model.get_test_accuracy(test_X, test_y)

