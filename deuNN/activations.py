import theano
import theano.tensor as T

"""
activition functions
Implmented commonly used activation functions listed on 
http://cs231n.github.io/neural-networks-1/#actfun
"""

## NOTE: theano version of softmax does NOT handle the large number of X, which cause problems of driving gradients to NaN sometimes.
#def softmax(x):
#    return T.nnet.softmax(x)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def relu(x):
    return (x + abs(x)) / 2.

def tanh(x):
    return T.tanh(x)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)

def linear(x):
    return x

def leaky_relu(x):
    raise NotImplementedError

def maxout(x):
    raise NotImplementedError


# The module get function, used to fetch function handle from outside
# This is a basic routine in a module
from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'activations')
