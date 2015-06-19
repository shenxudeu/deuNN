import numpy as np
import os, sys
import cPickle
import time
import gzip

import theano
import theano.tensor as T

import pdb

from softmax_v2 import shared_data, shared_data_int32, to_categorical

def load_mnist(dataset="mnist.pkl.gz"):
    """
    dataset: string, the path to dataset (MNIST)
    """
    data_dir, data_file = os.path.split(dataset)
    
    # download MNIST if not found
    if not os.path.isfile(dataset):
        import urllib
        origin = (
                'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading MNIST from %s' % origin
        assert urllib.urlretrieve(origin, dataset)
            
    print "Loading Data ..."

    with gzip.open(dataset, 'rb') as handle:
        train_set, valid_set, test_set = cPickle.load(handle)
    
    rval = [(train_set[0],to_categorical(train_set[1])),
            (valid_set[0],to_categorical(valid_set[1])),
            (test_set[0], to_categorical(test_set[1]))]
    #train_X, train_y = shared_data(train_set[0]),shared_data_int32(to_categorical(train_set[1]))
    #valid_X, valid_y = shared_data(valid_set[0]),shared_data_int32(to_categorical(valid_set[1]))
    #test_X, test_y   = shared_data(test_set[0]),shared_data_int32(to_categorical(test_set[1]))
    #
    #rval = [(train_X, train_y), (valid_X, valid_y), (test_X, test_y)]

    return rval


batch_size = 500
n_epoch = 20
learning_rate = 0.13

datasets = load_mnist()

train_X, train_y = datasets[0]
valid_X, valid_y = datasets[1]
test_X, test_y   = datasets[2]

(N, D) = train_X.shape
k = 10

X = T.matrix(dtype='float32')
y = T.matrix(dtype='int32')

W = theano.shared(np.zeros((D,k),dtype=theano.config.floatX))
b = theano.shared(np.zeros((k,),dtype=theano.config.floatX))
params = [W,b]

py_x = T.nnet.softmax(T.dot(X,W)+b)
loss = T.mean(T.nnet.categorical_crossentropy(py_x,y))

grads = T.grad(loss, params)
updates = []
for p,g in zip(params, grads):
    updates.append((p, p- learning_rate * g))

train = theano.function(inputs=[X,y],
        outputs=loss, updates=updates, allow_input_downcast=True)

iter_num = 0
pdb.set_trace()
for i in xrange(n_epoch):
    for start, end in zip(range(0,N, batch_size), range(batch_size,N,batch_size)):
        loss_val = train(train_X[start:end],train_y[start:end])
        print '%d iteration, %d epoch, cost = %f'%(iter_num, i, loss_val)
        iter_num += 1














