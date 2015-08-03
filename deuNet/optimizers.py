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
        """
        Reference: http://www.magicbroom.info/Papers/DuchiHaSi10.pdf
        """
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
                updates.append((p,p + self.momentum * v - lr * g))
            else:
                #updates.append((p, p - lr * g))
                updates.append((p, p + v))

        return updates


class RMSprop(object):
    """
    RMSprop: a very effecitive, but unpublished adaptive learning rate method.
    It is introduced by Hinton in his Coursera class.
    Reference: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    def __init__(self,lr=0.01, epsilon = 1e-8, rho = 0.9):
        self.lr = lr
        self.epsilon = epsilon
        self.rho = rho

    def set_lr(self, lr = 0.01):
        self.lr = lr

    def set_rho(self, rho = 0.9):
        self.rho = rho

    def get_gradients(self, loss, params):
        grads = T.grad(loss, params)

        return grads

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        caches = [shared_zeros(p.get_value().shape) for p in params]
        updates = []

        for p,g,c in zip(params, grads, caches):
            new_c = self.rho * c + (1 - self.rho) * g ** 2
            updates.append((c,new_c))

            new_p = p - self.lr * g / T.sqrt(new_c + self.epsilon)
            updates.append((p, new_p))

        return updates


class Adagrad(object):
    """
    Adagrad: It is an adaptive learning rate method proposed by J. Duchi.
    It is very similar with RMSprop, but with no extra hyper-parameters
    Reference: http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """
    def __init__(self, lr=0.01, epsilon = 1e-8):
        self.lr = lr
        self.epsilon = epsilon

    def set_lr(self, lr = 0.01):
        self.lr = lr
    
    def get_gradients(self, loss, params):
        grads = T.grad(loss, params)

        return grads

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        caches = [shared_zeros(p.get_value().shape) for p in params]
        updates = []

        for p,g,c in zip(params, grads, caches):
            new_c = c + g **2
            updates.append((c, new_c))

            new_p = p - self.lr *g / T.sqrt(new_c + self.epsilon)
            updates.append((p, new_p))
        
            return updates


class Adadelta(object):
    """
    Adadelta: another relatively common adaptive learning rate method.
    Reference: http://arxiv.org/abs/1212.5701
    """
    def __init__(self, lr = 0.01, rho = 0.9, epsilon = 1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.rho = rho

    def set_lr(self, lr):
        self.lr = lr

    def set_rho(self, rho):
        self.rho = rho
    
    def get_gradients(self, loss, params):
        grads = T.grad(loss, params)

        return grads

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        caches = [shared_zeros(p.get_value().shape) for p in params]
        delta_caches = [shared_zeros(p.get_value().shape) for p in params]
        updates = []

        for p,g,c,dc in zip(params, grads, caches, delta_caches):
            # update caches
            new_c = self.rho * c + (1 - self.rho) * g ** 2
            updates.append((c,new_c))

            # update params
            update = g * T.sqrt(dc + self.epsilon) / T.sqrt(new_c + self.epsilon)

            new_p = p - self.lr * update
            updates.append((p, new_p))

            # update delta_caches
            new_dc = self.rho * dc + (1 - self.rho) * update ** 2
            updates.append((dc, new_dc))
        
        return updates


from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'optimizers', instantiate=True)
