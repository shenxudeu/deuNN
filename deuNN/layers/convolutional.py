import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from .. import activations, initializations
from .. utils.theano_utils import shared_zeros
from .. layers.core import Layer


class Convolution2D(Layer):
    def __init__(self, nb_filter, stack_size, nb_row, nb_col,
            init='glorot_uniform',activation='linear',weights=None,
            border_mode='valid',subsample=(1,1),
            W_regularizer=None, b_regularizer=None, activity_regularizer=None):

        if border_mode not in {'valid', 'full', 'same'}:
            raise Exception("Invalid boder mode for ConvNet 2D")

