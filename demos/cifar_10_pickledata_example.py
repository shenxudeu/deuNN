"""
Example of train a 2-layers Neural Network classifier on CIFAR-10 dataset
"""
import numpy as np
import sys, os
import cPickle

sys.path.append("../../deuNet/")

from deuNet.utils import np_utils
from deuNet.datasets import cifar10
from deuNet.models import NN
from deuNet.layers.core import AffineLayer, Dropout
from deuNet.layers.convolutional import Convolution2D,Flatten,MaxPooling2D
from deuNet.layers.batch_normalization import BatchNormalization,LRN2D

import pdb
np.random.seed(1984)

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

N = len(train_X)

# Generate cPickle files seperate the dataset
def seg_data(X,y,fn_prefix,seg_size):
    N = len(X)
    seg_size = 500
    fid = 0
    for start, end in zip(range(0,N,seg_size), range(seg_size,N,seg_size)):
        tmp = [X[start:end], y[start:end]]
        fn = '%s_%d.plk'%(fn_prefix,fid)
        with open(fn,'wb') as f:
            cPickle.dump(tmp, f)
            fid += 1
try:
    os.mkdir('cifar10SegData')
except:
    pass
#seg_data(train_X,train_y, 'cifar10SegData/train',500)
#seg_data(valid_X,valid_y, 'cifar10SegData/valid',500)

#fn_list = list('cifar10SegData/train_%d.plk'%i for i in range(0,99))
#for train_data in np_utils.DataIterator(fn_list,batch_size):
#    print train_data[0].shape
#    print train_data[0][0,:10]

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
fn_list = list('cifar10SegData/train_%d.plk'%i for i in range(0,99))
trainIterator = np_utils.DataIterator(fn_list,batch_size)

fn_list = list('cifar10SegData/valid_%d.plk'%i for i in range(0,19))
validIterator = np_utils.DataIterator(fn_list,batch_size)

model.fit_iterator(trainIterator, validIterator, N, 
        batch_size=batch_size, nb_epoch=nb_epoch, verbose=True,
        epoch_step=epoch_step,lr_drop_rate=lr_drop_rate)

# Test NN
model.get_test_accuracy(test_X, test_y)

