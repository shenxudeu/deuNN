import numpy as np
import sys, os
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time
import pdb
np.random.seed(1337)


sys.path.append('../../deuNN/')
from deuNN.datasets import mnist
from deuNN.utils import np_utils
from deuNN.utils.theano_utils import sharedX, shared_zeros
from deuNN.optimizers import SGD

srng = RandomStreams(seed=np.random.randint(10e6))

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
    np.random.seed(1337)
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

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g **2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))

    return updates


def model(X, w, w1, w2, b2, w_o, b_o, p_drop_conv, p_drop_hidden):
    l1 = rectify(conv2d(X, w, border_mode = 'full'))
    l2a = rectify(conv2d(l1, w1, border_mode = 'valid'))
    l2  = max_pool_2d(l2a,(2,2))
    #l2  = dropout(l2, p_drop_conv)

    l3  = T.flatten(l2, outdim = 2)
    l3  = rectify(T.dot(l3,w2)+b2)
    #l3  = dropout(l3, p_drop_hidden)
    
    pyx  = softmax(T.dot(l3,w_o)+b_o)

    return l1, l2, l3, pyx


#[train_X, train_y, valid_X, valid_y, test_X, test_y] = mnist.load_data()
#[train_set, valid_set, test_set] = mnist.load_data()
#(train_X, train_y) = train_set
#(valid_X, valid_y) = valid_set
#(test_X, test_y) = test_set
#train_X = np.vstack((train_X,valid_X))
#train_y = np.hstack((train_y,valid_y))
#valid_X, valid_y = test_X, test_y

(train_X, train_y),(test_X, test_y) = mnist.load_data()
(train_X,train_y), (test_X,test_y) = mnist.load_data()
valid_X,valid_y = test_X, test_y
nb_classes = 10

train_X = train_X.reshape(train_X.shape[0], 1, 28, 28)
valid_X = valid_X.reshape(valid_X.shape[0], 1, 28, 28)
test_X = test_X.reshape(test_X.shape[0], 1, 28, 28)

train_y = np_utils.one_hot(train_y, nb_classes)
valid_y = np_utils.one_hot(valid_y, nb_classes)
test_y = np_utils.one_hot(test_y, nb_classes)

train_X = train_X.astype("float32")
valid_X = valid_X.astype("float32")
test_X = test_X.astype("float32")
train_X /= 255
valid_X /= 255
test_X /= 255

X = T.ftensor4()
y = T.fmatrix()
w = glorot_uniform((32,1,3,3))
w1 = glorot_uniform((32,32,3,3))
w2 = glorot_uniform((32*196,128))
w_o = glorot_uniform((128,10))
b2 = shared_zeros(128)
b_o = shared_zeros(10)

noise_l1, noise_l2, noise_l3, noise_py_x = model(X, w, w1, w2,b2, w_o, b_o, 0.25, 0.5)

l1, l2, l3, py_x = model(X, w, w1, w2, b2, w_o, b_o, 0.25, 0.5)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, y))
params = [w, w1, w2, b2,w_o, b_o]
optimizer = SGD(lr = 0.01, momentum = 0., decay = 0, nesterov=False)
updates = optimizer.get_updates(cost, params)
#updates = RMSprop(cost, params, lr = 0.01)

train_accuracy = T.mean(T.eq(T.argmax(y, axis=-1), T.argmax(py_x, axis=-1)))

print "Compile Network"
train = theano.function(
        inputs = [X, y],
        outputs = cost,
        updates = updates, allow_input_downcast = True)

train_acc = theano.function(
        inputs = [X, y],
        outputs = [cost, train_accuracy],
        updates = updates, allow_input_downcast = True)


predict = theano.function(
        inputs = [X],
        outputs = y_x,
        allow_input_downcast = True)

print "Start Training"
num_iter = 0
show_frequency = 100
for i in xrange(100):
    for start, end in zip(range(0,len(train_X),128),range(128,len(train_X),128)):
        #cost = train(train_X[start:end], train_y[start:end])
        [cost,tr_acc] = train_acc(train_X[start:end], train_y[start:end])
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


