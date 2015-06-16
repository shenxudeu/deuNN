import numpy as np
import os, sys
import cPickle
import time
import gzip

import theano
import theano.tensor as T

class SoftMax(object):
    def __init__(self, input, n_in, n_out):
        """
        input: theano var, N x D
        n_int: D
        n_out: k

        weights:
            W: theano var, D x k
            b: theano var, k
        """
        # Initialize weights
        self.W = theano.shared(
                np.zeros((n_in, n_out),
                    dtype=theano.config.floatX),
                name='W', borrow = True)

        self.b = theano.shared(
                np.zeros((n_out,),
                    dtype=theano.config.floatX),
                name='b', borrow = True)
        
        # Forward Propagation
        # probs by softmax function
        self.p_y_given_x = T.nnet.softmax(T.dot(W,input)+self.b) # (N x k)
        # output class
        self.y_pred  = T.argmax(self.p_y_given_x,axis=1)

        self.params = [self.W, self.b]

    def loss(self, y):
        data_loss = -T.log(self.p_y_given_x)[T.arange(y.shape[0]),y]
        return data_loss

    def errors(self, y):
        return T.mean(T.neq(y, self.y_pred))


def load_mnist(dataset):
    """
    dataset: string, the path to dataset (MNIST)
    """
    data_dir, data_file = os.path.split(dataset)
    
    # download MNIST if not found
    if not os.path.isfile(dataset):
        import urllib
        orgin = (
                'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading MNIST from %s' % origin
        assert urllib.urlretrieve(origin, dataset)
            
    print "Loading Data ..."

    with gzip.open(dataset, 'rb') as handle:
        train_set, valid_set, test_set = cPickle.load(handle)
    
    def shared_dataset(data_xy,borrow = True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                 dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                 dtype=theano.config.floatX), borrow=borrow)

    train_X, train_y = shared_dataset(train_set)
    valid_X, valid_y = shared_dataset(valid_set)
    test_X, test_y   = shared_dataset(test_set)

    rval = [(train_X, train_y), (valid_X, valid_y), (test_X, test_y)]

    return rval


def train_sgd(learning_rate = .13, n_epochs=1000,
        dataset='mnist.pkl.gz', batch_size = 600):
    datasets = load_mnist(dataset)

    train_X, train_y = datasets[0]
    valid_X, valid_y = datasets[1]
    test_X, test_y   = datasets[2]

    n_train_batches = train_X.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_X.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_X.get_value(borrow=True).shape[0] / batch_size




