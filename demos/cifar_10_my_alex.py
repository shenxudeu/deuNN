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
from deuNet.layers.batch_normalization import BatchNormalization,LRN2D
from deuNet.preprocessing.image import ImageDataGenerator

import pdb
np.random.seed(1984)

DATA_AUGMENTATION = True

batch_size = 128
nb_classes = 10
nb_epoch = 100
learning_rate = 0.01
w_scale = 1e-2
momentum = 0.9
lr_decay = 1e-7
w_decay = 5e-4
epoch_step = 10
lr_drop_rate = 0.1
nesterov = True
rho = 0.9
reg_W = 0.

checkpoint_fn = '.trained_cifar10_cnn.h5'
if DATA_AUGMENTATION:
    log_fn = '.cifar10_myAlex_aug.log'
else:
    log_fn = '.cifar10_myAlex.log'

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


pool  = lambda x, y, kernel, stride: ((x-kernel)/stride+1, (y-kernel)/stride+1) 

def ConvSame(nInputFilters,nOutputFilters, model):
    model.add(Convolution2D(nOutputFilters,nInputFilters,5,5, border_mode='full',
        init='glorot_uniform',activation='relu', reg_W=reg_W))
    model.add(Convolution2D(nOutputFilters,nOutputFilters,5,5, border_mode='valid',
        init='glorot_uniform',activation='relu', reg_W=reg_W))
    return model

def ConvSameBN(nInputFilters,nOutputFilters, model):
    model.add(Convolution2D(nOutputFilters,nInputFilters,5,5, border_mode='full',
        init='glorot_uniform',activation='linear', reg_W=reg_W))
    model.add(BatchNormalization((1,nOutputFilters,1,1),activation='relu'))
    model.add(Convolution2D(nOutputFilters,nOutputFilters,5,5, border_mode='valid',
        init='glorot_uniform',activation='linear', reg_W=reg_W))
    model.add(BatchNormalization((1,nOutputFilters,1,1),activation='relu'))
    return model


ignore_border = True

# NN architecture
model = NN(checkpoint_fn,log_fn)

nh, nw = (32, 32)

BN = False
if BN:
    ConvS = ConvSameBN
else:
    ConvS = ConvSame

model = ConvS(3,32, model)
model.add(MaxPooling2D(pool_size=(3,3),stride=(2,2),ignore_border=ignore_border))
nh, nw = pool(nh, nw, 3,2)
model.add(LRN2D(n=3))
model.add(Dropout(0.25, uncertainty=False)) 

model = ConvS(32,32, model)
model.add(MaxPooling2D(pool_size=(3,3),stride=(2,2),ignore_border=ignore_border))
nh, nw = pool(nh, nw, 3,2)
model.add(LRN2D(n=3))
model.add(Dropout(0.25, uncertainty=False)) 

model = ConvS(32,64, model)
model.add(MaxPooling2D(pool_size=(3,3),stride=(2,2),ignore_border=ignore_border))
nh, nw = pool(nh, nw, 3,2)
model.add(LRN2D(n=3))
model.add(Dropout(0.25, uncertainty=False)) 


model.add(Flatten())
model.add(AffineLayer(nh*nw*64, 512,activation='relu',reg_W=reg_W, init='glorot_uniform'))
model.add(Dropout(0.5, uncertainty=False))
model.add(AffineLayer(512, nb_classes,activation='softmax',reg_W=reg_W,init='glorot_uniform'))


# Compile NN
print 'Compile NN ...'
model.compile(optimizer='SGD', loss='categorical_crossentropy',
        reg_type='L2', learning_rate = learning_rate, momentum=momentum,
        lr_decay=lr_decay, nesterov=nesterov, rho=rho, w_decay=w_decay)

# Train NN
if not DATA_AUGMENTATION:
    model.fit(train_X, train_y, valid_X, valid_y,
            batch_size=batch_size, nb_epoch=nb_epoch, verbose=True,
            epoch_step=epoch_step,lr_drop_rate=lr_drop_rate)

else:
    print "Using Data Augmentation"
    datagen = ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=False,
            featurewise_std_normalization=True,
            samplewise_std_normalization=False,
            
            zca_whitening=False,
            rotation_range = 20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False)
    
    datagen.fit(train_X)

    trainIterator = datagen.flow(train_X, train_y, batch_size=batch_size,transform=True)
    validIterator = datagen.flow(valid_X, valid_y, batch_size=batch_size,transform=False)
    
    model.fit_iterator(trainIterator, validIterator, len(train_X),
            batch_size=batch_size, nb_epoch=nb_epoch, verbose=True,
            epoch_step=epoch_step,lr_drop_rate=lr_drop_rate)


# Test NN
model.get_test_accuracy(test_X, test_y)

