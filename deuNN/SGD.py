import theano
import theano.tensor as T
import numpy as np

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
    def __init__(self, lr=0.01):
        self.lr = lr
        self.iterations = shared_scalar(0.)
    
    def set_lr(self, lr=0.01):
        self.lr = lr

    def get_gradients(self, loss, params):
        grads = T.grad(loss, params)

        return grads

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        updates = [(self.iterations, self.iterations+1.)]

        for p, g in zip(params, grads):
            updates.append((p, p - self.lr * g))

        return updates
        

from .utils.generic_utils import get_from_modules
def get(identifier):
    return get_from_modules(identifier, globals(), 'SGD', instantiate=True)
