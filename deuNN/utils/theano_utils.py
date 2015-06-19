import theano
import theano.tensor as T
import numpy as np

"""
theano_utils
Commonly used functions to convert np array to theano.shared variable
theano.shared variable is used to store data into GPUs
"""

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X,dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X,dtype=dtype), name=name)

def shared_zeros(shape,dtype=theano.config.floatX,name=None):
    return theano.shared(np.asarray(np.zeros(shape), dtype=dtype), name=name)

def shared_ones(shape, dtype=theano.config.floatX,name=None):
    return theano.shared(np.asarray(np.ones(shape), dtype=dtype), name=name)

def shared_scalar(val=0., dtype=theano.config.floatX,name=None):
    return theano.shared(np.cast[dtype](val))

