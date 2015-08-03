import numpy as np
import sys, os
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

sys.path.append('../../deuNet/')
from deuNet.datasets import cifar10
from deuNet.utils import np_utils
from deuNet.utils.theano_utils import sharedX, shared_zeros
from deuNet.optimizers import SGD, RMSprop

import pdb
np.random.seed(1984)

srng = RandomStreams()

import pdb

def shared_data(np_data, borrow=True):
    return theano.shared(np.asarray(np_data, dtype=theano.config.floatX), borrow=borrow)
def shared_data_int32(np_data, borrow=True):
    return theano.shared(np.asarray(np_data, dtype='int32'), borrow=borrow)

def floatX(X):
    return np.asarray(X,dtype=theano.config.floatX)

def get_fans(shape):
    fan_in = shape[0] if len(shape)==2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

def uniform(shape, scale=1e-5):
    np.random.seed(1984)
    return sharedX(np.random.uniform(low=-scale,high=scale,size=shape))

def glorot_uniform(shape, scale=None):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6./(fan_in+fan_out))
    return uniform(shape, s)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p= 0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

#def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
#    grads = T.grad(cost=cost, wrt=params)
#    updates = []
#    for p, g in zip(params, grads):
#        acc = theano.shared(p.get_value() * 0.)
#        acc_new = rho * acc + (1 - rho) * g **2
#        gradient_scaling = T.sqrt(acc_new + epsilon)
#        g = g / gradient_scaling
#        updates.append((acc, acc_new))
#        updates.append((p, p - lr * g))
#
#    return updates


def model(X, w, b, w_o, b_o):
    #l1 = rectify(conv2d(X, w, border_mode = 'full'))
    
    #l2a = rectify(conv2d(l1, w1, border_mode = 'valid'))
    #l2  = max_pool_2d(l2a,(2,2))
    #l2  = dropout(l2, p_drop_conv)

    #l3 = rectify(conv2d(l2, w2, border_mode = 'full'))
    
    #l4a = rectify(conv2d(l3, w3, border_mode = 'valid'))
    #l4  = max_pool_2d(l4a,(2,2))
    #l4  = dropout(l4, p_drop_conv)

    #l1  = T.flatten(X, outdim = 2)
    #l2  = rectify(T.dot(X,w)+b)
    l2  = T.dot(X,w)+b
    #l5  = dropout(l5, p_drop_hidden)
    
    pyx  = softmax(T.dot(l2,w_o)+b_o)

    return  l2, pyx


#[train_X, train_y, valid_X, valid_y, test_X, test_y] = cifar_10.load_data()
(train_X,train_y), (test_X,test_y) = cifar10.load_data()
valid_X,valid_y = test_X, test_y
nb_classes = 10
 
nb_channels, nb_w, nb_h = train_X.shape[1], train_X.shape[2], train_X.shape[3]
train_X = train_X.reshape(-1,nb_channels*nb_w*nb_h).astype('float32')
valid_X = valid_X.reshape(-1,nb_channels*nb_w*nb_h).astype('float32')
test_X = test_X.reshape(-1,nb_channels*nb_w*nb_h).astype('float32')
D = train_X.shape[1]

train_y = np_utils.one_hot(train_y, nb_classes)
valid_y = np_utils.one_hot(valid_y, nb_classes)
test_y = np_utils.one_hot(test_y, nb_classes)
#train_X = train_X.astype("float32")
#valid_X = valid_X.astype("float32")
train_X /= 255
valid_X /= 255
test_X /= 255

#X = T.ftensor4()
X = T.fmatrix()
y = T.fmatrix()
w = glorot_uniform((32*32*3,32*32))
w_o = glorot_uniform((32*32,10))
b = shared_zeros(32*32)
b_o = shared_zeros(10)

noise_l2,noise_py_x = model(X, w, b, w_o, b_o)

l2, py_x = model(X, w, b, w_o, b_o)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, y))
params = [w, b, w_o, b_o]
#optimizer = SGD(lr = 1e-2, momentum = None, decay = None, nesterov=False)
optimizer = RMSprop(lr=1e-2,rho=0.9)
updates = optimizer.get_updates(cost, params)
#updates = RMSprop(cost, params, lr = 0.01)

print "Compile Network"
train = theano.function(
        inputs = [X, y],
        outputs = cost,
        updates = updates, allow_input_downcast = True)

predict = theano.function(
        inputs = [X],
        outputs = y_x,
        allow_input_downcast = True)

grads = optimizer.get_gradients(cost, params)
get_grads = theano.function(
        inputs = [X, y],
        outputs = grads,
        allow_input_downcast=True)

get_prob = theano.function(
        inputs = [X],
        outputs = noise_py_x,
        allow_input_downcast=True)

print "Start Training"
num_iter = 0
show_frequency = 100
for i in xrange(100):
    for start, end in zip(range(0,len(train_X),100),range(100,len(train_X),100)):
        cost_val = train(train_X[start:end], train_y[start:end])
        grads_val = get_grads(train_X[start:end], train_y[start:end])
        prob_val = get_prob(train_X[start:end])
        pdb.set_trace()
        num_iter += 1
        if num_iter % show_frequency == 0:
            print 'Iteration %d, epoch %d: cost = %f'%(num_iter,i,cost)
    #train_acc = np.mean(np.argmax(train_y,axis=1) == predict(train_X))
    valid_acc = np.mean(np.argmax(valid_y,axis=1) == predict(valid_X))
    #print 'Training Acc. %f, Validation Acc. %f'%(train_acc, valid_acc)
    print '----Validation Acc. %f'%(valid_acc)

test_acc = np.mean(np.argmax(test_y,axis=1) == predict(test_X))
print '*** Test Acc .%f'%test_acc


