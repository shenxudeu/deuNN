"""
Example of train a 2-layers Neural Network classifier on CIFAR-10 dataset
"""
import numpy as np
import sys

sys.path.append("../../deuNet/")

from deuNet.utils import np_utils
from deuNet.datasets import cifar10
from deuNet.datasets import mnist
from deuNet.models import NN
from deuNet.layers.core import AffineLayer, Dropout
from deuNet.layers.convolutional import Convolution2D,Flatten,MaxPooling2D
from deuNet.layers.batch_normalization import BatchNormalization

import pdb
np.random.seed(1984)

batch_size = 128
nb_classes = 10
nb_epoch = 100
learning_rate = 1.
w_scale = 1e-2
momentum = 0.9
lr_decay = 1e-7
nesterov = False
rho = 0.9
reg_W = 0.

checkpoint_fn = '.trained_cifar10_vgg.h5'
log_fn = '.cifar10_vgg.log'

#(train_X, train_y), (test_X, test_y) = cifar10.load_data()
#valid_X,valid_y = test_X, test_y
#
#n_batchs = int(train_X.shape[0] / batch_size)
#
## convert data_y to one-hot
#train_y = np_utils.one_hot(train_y, nb_classes)
#valid_y = np_utils.one_hot(valid_y, nb_classes)
#test_y = np_utils.one_hot(test_y, nb_classes)
#
#train_X = train_X.astype("float32")
#valid_X = valid_X.astype("float32")
#test_X = test_X.astype("float32")
#train_X /= 255
#valid_X /= 255
#test_X  /= 255

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



# size calculation
valid = lambda x, y, kernel, stride: ((x-kernel)/stride+1, (y-kernel)/stride+1)
full  = lambda x, y, kernel, stride: ((x+kernel)/stride-1, (y+kernel)/stride-1)
pool  = lambda x, y, kernel, stride: ((x-kernel)/stride+1, (y-kernel)/stride+1)

#def ConvBNRelu(nOutputPlane, nInputPlane,model):
#    model.add(Convolution2D(nOutputPlane,nInputPlane,3,3, border_mode='same',
#        init='glorot_uniform',activation='linear', reg_W=reg_W))   
#    model.add(BatchNormalization((1,nOutputPlane,1,1),activation='relu'))
#    return model

# NN architecture
model = NN(checkpoint_fn)
nh, nw = (32, 32)

if 0:
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



#model = ConvBNRelu(64, 1, model)
nOutputPlane,nInputPlane = 32, 1
model.add(Convolution2D(nOutputPlane,nInputPlane,4,4, border_mode='same',
	init='glorot_uniform',activation='linear', reg_W=reg_W))   
model.add(BatchNormalization((1,32,1,1),activation='relu'))

#model.add(Dropout(0.3, uncertainty=False))
#
#model = ConvBNRelu(64, 64, model)
#model.add(MaxPooling2D(pool_size=(2,2),ignore_border=True))
#nh, nw = pool(nh, nw, 2,2)
#
#model = ConvBNRelu(128, 64, model)
#model.add(Dropout(0.4, uncertainty=False))
#
#model = ConvBNRelu(128, 128, model)
#model.add(MaxPooling2D(pool_size=(2,2),ignore_border=True))
#nh, nw = pool(nh, nw, 2,2)
#
#model = ConvBNRelu(256, 128, model)
#model.add(Dropout(0.4, uncertainty=False))
#
#model = ConvBNRelu(256, 256, model)
#model.add(Dropout(0.4, uncertainty=False))
#
#model = ConvBNRelu(256, 256, model)
#model.add(MaxPooling2D(pool_size=(2,2),ignore_border=True))
#nh, nw = pool(nh, nw, 2,2)
#
#model = ConvBNRelu(512, 256, model)
#model.add(Dropout(0.4, uncertainty=False))
#
#model = ConvBNRelu(512, 512, model)
#model.add(Dropout(0.4, uncertainty=False))
#
#model = ConvBNRelu(512, 512, model)
#model.add(MaxPooling2D(pool_size=(2,2),ignore_border=True))
#nh, nw = pool(nh, nw, 2,2)
#
#model = ConvBNRelu(512, 512, model)
#model.add(Dropout(0.4, uncertainty=False))
#
#model = ConvBNRelu(512, 512, model)
#model.add(Dropout(0.4, uncertainty=False))
#
#model = ConvBNRelu(512, 512, model)
#model.add(MaxPooling2D(pool_size=(2,2),ignore_border=True))
#nh, nw = pool(nh, nw, 2,2)
#
#
model.add(Flatten())
model.add(Dropout(0.5,uncertainty=False))
model.add(AffineLayer(nh*nw*512, 512,activation='linear',reg_W=reg_W, init='glorot_uniform'))
model.add(BatchNormalization((1,512),activation='relu'))
model.add(Dropout(0.5, uncertainty=False))
model.add(AffineLayer(512, nb_classes,activation='softmax',reg_W=reg_W,init='glorot_uniform'))


# Compile NN
print 'Compile NN ...'
model.compile(optimizer='SGD', loss='categorical_crossentropy',
        reg_type='L2', learning_rate = learning_rate, momentum=momentum,
        lr_decay=lr_decay, nesterov=nesterov, rho=rho,decay_freq=25,n_batchs=1000)

# Train NN
model.fit(train_X, train_y, valid_X, valid_y,
        batch_size=batch_size, nb_epoch=nb_epoch, verbose=True)

# Test NN
model.get_test_accuracy(test_X, test_y)

