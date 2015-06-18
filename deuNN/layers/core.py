import theano
import theano.tensor as T

from ..utils.theano_utils import shared_zeros, floatX

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
        self.param = []



class FullyConnectedLayer(Layer):
    """
    fully connected layer
    """
    def __init__(self, nb_input, nb_output, init='uniform',
            activation='linear', W_regularizer=None):
        super(FullyConnectedLayer, self).__init__()
        #self.init = 
