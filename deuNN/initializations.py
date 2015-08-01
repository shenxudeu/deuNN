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

def get_fans(shape):
    fan_in = shape[0] if len(shape)==2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

def uniform(shape, scale=1e-5):
    np.random.seed(1337)
    return sharedX(np.random.uniform(low=-scale,high=scale,size=shape))

def normal(shape, scale=1e-5):
    return sharedX(np.random.randn(*shape) * scale)

def lecun_uniform(shape, scale=None):
    """
    LeCun introduced this on 1998, Efficient Backprop
    http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(3./fan_in)
    return uniform(shape, scale)

def glorot_normal(shape, scale=None):
    """
    Introduced by Glorot and Bengio at 2010.
    Reference: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2./(fan_in+fan_out))
    return normal(shape, s)

def glorot_uniform(shape, scale=None):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6./(fan_in+fan_out))
    return uniform(shape, s)

def he_normal(shape, scale=None):
    """
    Introduced by Kaiming He and etc on 2015. 
    Reference: http://arxiv.org/abs/1502.01852
    """
    fan_in,fan_out = get_fans(shape)
    s = np.sqrt(2./fan_in)
    return normal(shape, s)

def he_uniform(shape, scale=None):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6./fan_in)
    return uniform(shape, s)


from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'initializations')
