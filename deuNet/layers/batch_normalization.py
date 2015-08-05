import theano
import theano.tensor as T

from .. import activations, initializations
from .. utils.theano_utils import shared_zeros
from .. layers.core import Layer

class BatchNormalization(Layer):
    """
    Batch Normalization:
    Published by S. Ioffee and C. Szegedy from Google.Solve the gradient
    divergence and vanish problem by normalizing features before non-linear activations.

    http://arxiv.org/pdf/1502.03167v3.pdf
    """
    def __init__(self, input_shape, epsilon=1e-6,
            momentum=0.9):
        super(BatchNormalization,self).__init__()
        self.init = initializations.get("uniform")
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.momentum = momentum

        self.gamma = self.init((self.input_shape))
        self.beta = shared_zeros(self.input_shape)

        self.running_mean = None
        self.beta = None
        

