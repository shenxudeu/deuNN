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

class Optimizer(object):
    """
    abstract object: graident descent optimization method
    """
    def get_gradients(self, data_loss, params, regularizers, regs):
        """
        Compute gradients
        Inputs:
            - data_loss: tensor.scalar, data loss
            - params: list(tensor.matrix/vector), weights and bias on each layer
            - regularizers: list(regularizer functions), used to compute the
                regularization term
            - regs: list of float32, regularizer strengths
        Outputs:
            - grads: list(tensor.matrix/vector), gradients of weights and bias on
                each layer
        Notes: params list and regularizers list has the same size, they contain
            all the parameters on each layer
        """
        #r_loss = [reg * reg_handle for (reg, reg_handle) in zip(regs, regularizers)]
        #total_loss = [(d_loss+ reg * reg_handle()) for ]
        #grads = T.grad
        def get_upates(self)




