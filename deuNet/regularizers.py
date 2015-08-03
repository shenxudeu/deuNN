import theano
import theano.tensor as T
import numpy as np

"""
regularization functions
commonly used L1 and L2
"""

def L1(x):
    return T.sum(abs(x))

def L2(x):
    return T.sum(x**2)


from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'regularizers')
