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

def categorical_crossentropy(y_probs, y_true):
    """
    calculate cross entropy loss
    Inputs:
        - y_probs: tensor.matrix, each row is a distribution(prob value)
        - y_true: tensor.matrix, one-hot presentation,
    Outputs:
        - tensor.scalar, data loss
    """
    return T.mean(T.nnet.categorical_crossentropy(y_probs, y_true))


from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'losses')

