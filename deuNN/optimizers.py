import theano
import theano.tensor as T
import numpy as np
from .utils.theano_utils import shared_scalar, shared_zeros

"""
# Optimizers: Gradient Computation and Parameter Updates
Once the analytic graident is computed with backpropagation (use theano),
the gradient are used to perform a parameter update.

I implementated several approches for performing the update in this module.

Reference: http://cs231n.github.io/neural-networks-3/#sgd
"""

class SGD(object):
    """
    abstract object: stocastic graident descent optimization method
    """
    def __init__(self, lr=0.01, momentum=None,decay=None,nesterov=None):
        self.lr = lr
        self.iterations = shared_scalar(0.)
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov
    
    def set_lr(self, lr=0.01):
        self.lr = lr

    def set_momentum(self, momentum=0.9):
        self.momentum = momentum

    def set_lr_decay(self, decay=0.99):
        self.decay = decay

    def set_nesterov(self, nesterov=True,momentum=0.9):
        self.momentum = momentum
        self.nesterov = nesterov
        if nesterov and momentum is None:
            raise ValueError('Use Nesterov, you must set momentum value')

    def get_gradients(self, loss, params):
        grads = T.grad(loss, params)

        return grads

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        if self.decay is not None:
            lr = self.lr * (1. / (1. + self.decay * self.iterations))
        else:
            lr = self.lr
        updates = [(self.iterations, self.iterations+1.)]

        for p, g in zip(params, grads):
            if self.momentum is not None:
                m = shared_zeros(p.get_value().shape)
                v = self.momentum * m - lr * g
                updates.append((m, v))
            else:
                v = -lr * g
            
            if self.nesterov:
                updates.append(p + self.momentum * v - lr * g)
            else:
                #updates.append((p, p - lr * g))
                updates.append((p, p + v))

        return updates
        

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'optimizers', instantiate=True)
