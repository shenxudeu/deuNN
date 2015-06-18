import numpy as np
import os, sys
import cPickle
import time
import gzip

import theano
import theano.tensor as T

import pdb

def shared_data(np_data, borrow=True):
    return theano.shared(np.asarray(np_data, dtype=theano.config.floatX), borrow=borrow)
def shared_data_int32(np_data, borrow=True):
    return theano.shared(np.asarray(np_data, dtype='int32'), borrow=borrow)


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
    
    train_X, train_y = shared_data(train_set[0]),shared_data_int32(to_categorical(train_set[1]))
    valid_X, valid_y = shared_data(valid_set[0]),shared_data_int32(to_categorical(valid_set[1]))
    test_X, test_y   = shared_data(test_set[0]),shared_data_int32(to_categorical(test_set[1]))
    
    rval = [(train_X, train_y), (valid_X, valid_y), (test_X, test_y)]

    return rval


def to_categorical(y, nb_classes=None):
    """
    convert class vector (int: 0 to num_classes)
    to one-hot presentation
    Input:
        - y: np array, N x 1
        - nb_classes: int, number of classes
    Output:
        - Y: theano.shared(dtype='int32'), one-hot presentation
    """
    N = len(y)
    y = np.asarray(y, dtype='int32')
    if nb_classes is None:
        nb_classes = 1 + np.max(y)
    Y = np.zeros((N,nb_classes))
    Y[np.arange(N),y] = 1.
    return Y


def train_softmax(learning_rate = .13, n_epochs=20,
        dataset='mnist.pkl.gz', batch_size = 500):
    
    # prepare dataset
    datasets = load_mnist(dataset)

    train_X, train_y = datasets[0]
    valid_X, valid_y = datasets[1]
    test_X, test_y   = datasets[2]
    n_train_batches = train_X.shape.eval()[0] / batch_size
    n_valid_batches = valid_X.shape.eval()[0] / batch_size
    n_test_batches = test_X.shape.eval()[0] / batch_size


    N, D = train_X.shape.eval()
    k = 10

    print "training data shape = %d x %d"%(N,D)
 
    # Build model
    # Initialize weights
    W = theano.shared(
            np.zeros((D, k),
                dtype=theano.config.floatX),
            name='W', borrow = True)
    b = theano.shared(
            np.zeros((k,),
                dtype=theano.config.floatX),
            name='b', borrow = True)
    X = T.matrix('x',dtype=theano.config.floatX)
    y = T.matrix('y',dtype='int32')

    # forward propagation
    p_y_given_x = T.nnet.softmax(T.dot(X,W)+b) # (N x k)
    data_loss = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, y))
    pred_error = T.mean(T.neq(T.argmax(y,axis=-1), T.argmax(p_y_given_x,axis=-1)))
    
    # backward propagation
    grad_W = T.grad(cost=data_loss, wrt=W)
    grad_b = T.grad(cost=data_loss, wrt=b)
    # update function
    updates = [(W,W - learning_rate * grad_W ),
               (b, b - learning_rate * grad_b)]
    
    # theano functions
    print "Compile model"
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
            outputs=1. - pred_error,
            givens={
                X: test_X[index*batch_size : (index+1)*batch_size],
                y: test_y[index*batch_size : (index+1)*batch_size]
            }
    )

    get_valid_acc = theano.function(
            inputs=[index],
            outputs=1. - pred_error,
            givens={
                X: valid_X[index*batch_size : (index+1)*batch_size],
                y: valid_y[index*batch_size : (index+1)*batch_size]
            }
    )
    
    get_train_acc = theano.function(
            inputs=[index],
            outputs=1. - pred_error,
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
    
if __name__ == '__main__':
    train_softmax(n_epochs=20)


