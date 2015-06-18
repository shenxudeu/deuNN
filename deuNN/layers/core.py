import theano
import theano.tensor as T

from ..utils.theano_utils import shared_zeros, floatX
from .. import activations, initializations

"""
# Core Layer Modual
 - A layer holds layer parameters, connections, regularizations,
   and forward-propagation computations.

 - Layer is the components container holds.

 - A Neurual Network architecture is a container holding various
   layers.  
"""

class Layer(object):
    """
    abstract class of layer
    """
    def __init__(self):
        self.params = []

    def get_input(self):
        if hasattr(self, 'previous'):
            return self.previous.get_output()
        else:
            return self.input

    def get_params(self):
        return self.params, self.regularizers


class AffineLayer(Layer):
    """
    Affine (fully connected) layer
    """
    def __init__(self, nb_input, nb_output, init='normal',
            activation='linear', W_regularizer=None):
        super(AffineLayer, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.nb_input = nb_input
        self.nb_output = nb_output
        
        # this symbolic variable will be used if this is the first layer
        self.input = T.matrix('input',dtype=theano.config.floatX)
        
        self.W = self.init((self.nb_input, self.nb_output))
        self.b = shared_zeros((self.nb_output))

        self.params = [self.W, self.b]

        self.regularizers = [W_regularizer]
    
    # forward pass for the affine layer
    def get_output(self):
        X = self.get_input()
        return self.activation(T.dot(X,self.W) + self.b)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'nb_input': self.nb_input,
                'nb_output': self.nb_output,
                'init': self.init.__name__,
                'activation': self.activation.__name__}
