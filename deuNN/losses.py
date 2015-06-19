import theano
import theano.tensor as T
import numpy as np

"""
# Losses: loss functions
In the training process, once we compute the output of a neurual network
  by input a training set (or mini-batch), in order to adjust the parameter,
  performance needs to be evaluated. classification accuracy is one measurement.
  However, accuracy is non-differentiable. Some better measurements have been 
  commonly used, such as cross-entropy for classification problem and L2-distance
  for regression problem.
"""

def categorical_crossentropy(py_x, y_true):
    """
    calculate cross entropy loss
    Inputs:
        - y_probs: tensor.matrix, each row is a distribution(prob value)
        - y_true: tensor.matrix, one-hot presentation,
    Outputs:
        - tensor.scalar, data loss
    """
    return T.mean(T.nnet.categorical_crossentropy(py_x, y_true))

def L1_regloss(params, regs):
    """
    compute regularization loss L1
    Inputs:
        - params: list of tensor vars, contains all the weights
        - regs: list of folat, contains reg strength of each weight
        Note: most times, the bias reg strength is set to 0.
    Outputs:
        - regloss: tensor scalar, the reg loss value
    """
    pass
    #for (p, reg) in zip(params, regs):



from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'losses')

