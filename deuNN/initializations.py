import theano
import theano.tensor as T
import numpy as np

from .utils.theano_utils import sharedX, shared_zeros
import pdb
"""
#Initializations
 - functions to return initialized variables
 - normally used to initialize weights
 - return a theano.shared variable
 - functions include http://cs231n.github.io/neural-networks-2/#init
"""

def uniform(shape, scale=1e-5):
    return sharedX(np.random.uniform(shape) * scale)

def normal(shape, scale=1e-5):
    return sharedX(np.random.randn(*shape) * scale)


from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'initializations')
