import numpy as np
import os, sys
import cPickle
import time
import gzip

import theano
import theano.tensor as T

import pdb

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
        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b) # (N x k)
        # output class
        self.y_pred  = T.argmax(self.p_y_given_x,axis=1)

        self.params = [self.W, self.b]

    def loss(self, y):
        data_loss = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
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
        origin = (
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
                                 dtype='int32'), borrow=borrow)
        return shared_x, shared_y
    train_X, train_y = shared_dataset(train_set)
    valid_X, valid_y = shared_dataset(valid_set)
    test_X, test_y   = shared_dataset(test_set)

    rval = [(train_X, train_y), (valid_X, valid_y), (test_X, test_y)]

    return rval


def train_sgd(learning_rate = .13, n_epochs=20,
        dataset='mnist.pkl.gz', batch_size = 500):
    
    # prepare dataset
    datasets = load_mnist(dataset)

    train_X, train_y = datasets[0]
    valid_X, valid_y = datasets[1]
    test_X, test_y   = datasets[2]
    n_train_batches = train_X.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_X.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_X.get_value(borrow=True).shape[0] / batch_size

    N, D = train_X.get_value().shape
    k = 10
    print "training data shape = %d x %d"%(N,D)
 
    # forward-backward propagation
    print "Building model"
    
    # forward propagation
    X = T.matrix('x')
    y = T.ivector('y')
    softmax = SoftMax(input=X, n_in=D, n_out=k)
    data_loss = softmax.loss(y)
    
    # backward propagation
    grad_W = T.grad(cost=data_loss, wrt=softmax.W)
    grad_b = T.grad(cost=data_loss, wrt=softmax.b)

    # update function
    updates = [(softmax.W,softmax.W - learning_rate * grad_W ),
               (softmax.b, softmax.b - learning_rate * grad_b)]
    
    # compile model, compile theano functions for train, validation, and test
    print "Compiling model"
    index = T.scalar(dtype='int32') #index to mini-batch
    train_model = theano.function(
            inputs=[index],
            outputs=data_loss,
            updates = updates,
            givens={
                X: train_X[index*batch_size : (index+1)*batch_size],
                y: train_y[index*batch_size : (index+1)*batch_size]
            }
    )

    get_test_acc = theano.function(
            inputs=[index],
            outputs=1. - softmax.errors(y),
            givens={
                X: test_X[index*batch_size : (index+1)*batch_size],
                y: test_y[index*batch_size : (index+1)*batch_size]
            }
    )

    get_valid_acc = theano.function(
            inputs=[index],
            outputs=1. - softmax.errors(y),
            givens={
                X: valid_X[index*batch_size : (index+1)*batch_size],
                y: valid_y[index*batch_size : (index+1)*batch_size]
            }
    )
    
    get_train_acc = theano.function(
            inputs=[index],
            outputs=1. - softmax.errors(y),
            givens={
                X: train_X[index*batch_size : (index+1)*batch_size],
                y: train_y[index*batch_size : (index+1)*batch_size]
            }
    )



    # mini-batch SGD
    print "Start training"
    start_time = time.clock()
    iter_num = 0
    acc_frequency = 200
    best_valid_acc = -np.inf
    for epoch in xrange(n_epochs):
        for minibatch_index in xrange(n_train_batches):
            iter_num += 1
            train_loss = train_model(minibatch_index)
            train_acc = np.mean([get_train_acc(i)
                                 for i in xrange(n_train_batches)])

            if iter_num % acc_frequency ==0:
                valid_acc = np.mean([get_valid_acc(i) for i in xrange(n_valid_batches)])

                print "Finished epoch %d / %d: cost %f, train: %f, val: %f, lr %e"%(
                        epoch, n_epochs, train_loss, train_acc, valid_acc, learning_rate)
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
    print "finished optimization, best validation accuracy: %f"%best_valid_acc
    end_time = time.clock()
    print "The training run for %d epochs, with %f epochs/sec"%(n_epochs,
            1.*n_epochs / (end_time - start_time))
    return softmax
    
if __name__ == '__main__':
    train_sgd()


