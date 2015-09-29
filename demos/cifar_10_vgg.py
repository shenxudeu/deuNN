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

batch_size = 128
nb_classes = 10
nb_epoch = 100
learning_rate = 1.5
w_scale = 1e-2
momentum = 0.9
lr_decay = 1e-7
nesterov = False
rho = 0.9
reg_W = 0.

checkpoint_fn = '.trained_cifar10_vgg.h5'
log_fn = '.cifar10_vgg.log'

(train_X, train_y), (test_X, test_y) = cifar10.load_data()
valid_X,valid_y = test_X, test_y

n_batchs = int(train_X.shape[0] / batch_size)

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

def ConvBNRelu(nOutputPlane, nInputPlane,model):
    model.add(Convolution2D(nOutputPlane,nInputPlane,3,3, border_mode='same',
        init='glorot_uniform',activation='linear', reg_W=reg_W))   
    model.add(BatchNormalization((1,nOutputPlane,1,1),activation='relu'))
    return model

# NN architecture
model = NN(checkpoint_fn)
nh, nw = (32, 32)
ignore_border = True
model = ConvBNRelu(64, 3, model)
model.add(Dropout(0.7, uncertainty=False))

model = ConvBNRelu(64, 64, model)
model.add(MaxPooling2D(pool_size=(2,2),ignore_border=ignore_border))
nh, nw = pool(nh, nw, 2,2)

model = ConvBNRelu(128, 64, model)
model.add(Dropout(0.6, uncertainty=False))

model = ConvBNRelu(128, 128, model)
model.add(MaxPooling2D(pool_size=(2,2),ignore_border=ignore_border))
nh, nw = pool(nh, nw, 2,2)

model = ConvBNRelu(256, 128, model)
model.add(Dropout(0.6, uncertainty=False))

model = ConvBNRelu(256, 256, model)
model.add(Dropout(0.6, uncertainty=False))

model = ConvBNRelu(256, 256, model)
model.add(MaxPooling2D(pool_size=(2,2),ignore_border=ignore_border))
nh, nw = pool(nh, nw, 2,2)

#model = ConvBNRelu(512, 256, model)
#model.add(Dropout(0.6, uncertainty=False))
#
#model = ConvBNRelu(512, 512, model)
#model.add(Dropout(0.6, uncertainty=False))
#
#model = ConvBNRelu(512, 512, model)
#model.add(MaxPooling2D(pool_size=(2,2),ignore_border=ignore_border))
#nh, nw = pool(nh, nw, 2,2)

#model = ConvBNRelu(512, 512, model)
#model.add(Dropout(0.4, uncertainty=False))
#
#model = ConvBNRelu(512, 512, model)
#model.add(Dropout(0.4, uncertainty=False))
#
#model = ConvBNRelu(512, 512, model)
#model.add(MaxPooling2D(pool_size=(2,2),ignore_border=ignore_border))
#nh, nw = pool(nh, nw, 2,2)


model.add(Flatten())
model.add(Dropout(0.5,uncertainty=False))
model.add(AffineLayer(nh*nw*256, 512,activation='linear',reg_W=reg_W, init='glorot_uniform'))
model.add(BatchNormalization((1,512),activation='relu'))
model.add(Dropout(0.5, uncertainty=False))
model.add(AffineLayer(512, nb_classes,activation='softmax',reg_W=reg_W,init='glorot_uniform'))


# Compile NN
print 'Compile NN ...'
model.compile(optimizer='SGD', loss='categorical_crossentropy',
        reg_type='L2', learning_rate = learning_rate, momentum=momentum,
        lr_decay=lr_decay, nesterov=nesterov, rho=rho,decay_freq=25,n_batchs=n_batchs)

# Train NN
model.fit(train_X, train_y, valid_X, valid_y,
        batch_size=batch_size, nb_epoch=nb_epoch, verbose=True)

# Test NN
model.get_test_accuracy(test_X, test_y)

