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
from deuNet.layers.batch_normalization import BatchNormalization

import pdb
np.random.seed(1984)

batch_size = 32
nb_classes = 10
nb_epoch = 100
learning_rate = 0.01
w_scale = 1e-2
momentum = 0.9
lr_decay = 0.1
nesterov = True
rho = 0.9
reg_W = 0.

checkpoint_fn = '.trained_cifar10_cnn.h5'

(train_X, train_y), (test_X, test_y) = cifar10.load_data()
valid_X,valid_y = test_X, test_y

n_batchs = train_X.shape[0]

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

# NN architecture
model = NN(checkpoint_fn)

model.add(Convolution2D(32,3,3,3, border_mode='full',
    init='glorot_uniform',activation='relu', reg_W=reg_W))
model.add(Convolution2D(32,32,3,3, border_mode='valid',
    init='glorot_uniform',activation='relu', reg_W=reg_W))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25, uncertainty=False))

model.add(Convolution2D(64,32,3,3, border_mode='full',
    init='glorot_uniform',activation='relu', reg_W=reg_W))
model.add(Convolution2D(64,64,3,3, border_mode='valid',
    init='glorot_uniform',activation='relu', reg_W=reg_W))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25,uncertainty=False))

model.add(Flatten())
model.add(AffineLayer(8*8*64, 512,activation='relu',reg_W=reg_W, init='glorot_uniform'))
model.add(Dropout(0.5, uncertainty=False))
model.add(AffineLayer(512, nb_classes,activation='softmax',reg_W=reg_W,init='glorot_uniform'))


# Compile NN
print 'Compile NN ...'
model.compile(optimizer='SGD', loss='categorical_crossentropy',
        reg_type='L2', learning_rate = learning_rate, momentum=momentum,
        lr_decay=lr_decay, nesterov=nesterov, rho=rho,decay_freq=5,n_batchs=n_batchs)

# Train NN
model.fit(train_X, train_y, valid_X, valid_y,
        batch_size=batch_size, nb_epoch=nb_epoch, verbose=True)

# Test NN
model.get_test_accuracy(test_X, test_y)

