import theano
import theano.tensor as T
import numpy as np

from .utils.theano_utils import shared_scalar

"""
# Losses: loss functions
In the training process, once we compute the output of a neurual network
  by input a training set (or mini-batch), in order to adjust the parameter,
  performance needs to be evaluated. classification accuracy is one measurement.
  However, accuracy is non-differentiable. Some better measurements have been 
  commonly used, such as cross-entropy for classification problem and L2-distance
  for regression problem.
"""

#######################################
## Loss Function for Classification  ##
#######################################
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


##################################
## Loss Function for Regression ##
##################################
def euclidean_loss(y_pred, y_true):
    """
    Compute euclidean loss on 1 dim
    Reference: http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html 
    section: Dropout and Deep Models
    """
    #return T.mean(T.abs_(T.max(y_pred,axis=1) - y_true))/2.
    return T.mean(T.abs_(y_pred - y_true))/2.

def mean_squared_error(y_pred, y_true):
    #return T.mean(T.sqrt(T.max(y_pred,axis=1) - y_true))
    return T.mean(T.sqrt(y_pred - y_true))

def mean_absolute_error(y_pred, y_true):
    """
    Almost the same as euclidean loss
    """
    #return T.mean(T.abs_(T.max(y_pred,axis=1) - y_true))
    return T.mean(T.abs_(y_pred - y_true))


#########################
## Regularization Loss ##
#########################
def L1(params, regs):
    """
    compute regularization loss L1
    Inputs:
        - params: list of tensor vars, contains all the weights
        - regs: list of folat, contains reg strength of each weight
        Note: most times, the bias reg strength is set to 0.
    Outputs:
        - regloss: tensor scalar, the reg loss value
    """
    reg_loss = shared_scalar(0.)
    for (p, reg) in zip(params, regs):
        reg_loss += reg * T.sum(abs(p))
    return reg_loss

def L2(params, regs):
    """
    compute L2 regularization loss
    """
    reg_loss = shared_scalar(0.)
    for (p, reg) in zip(params, regs):
        reg_loss += reg * T.sum(p**2)
    return reg_loss


from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'losses')

