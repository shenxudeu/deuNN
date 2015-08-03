import theano
import theano.tensor as T
import numpy as np

from ..utils.theano_utils import shared_zeros, floatX, shared_scalar
from .. import activations, initializations

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#srng = RandomStreams(seed=np.random.randint(10e6))
srng = RandomStreams()
import pdb

"""
# Core Layer Modual: The key component
    
    - get_output(): call get_input(), then compute forward pass and return layer output

    - get_input(): call previous-layer's forward function, this take cares of the real theano.graph construction

    - connect(): link-list like structure, set a pointer (called self.previous) point to last layer

    - get_params(): return the parameters for weights update

    Recurvise calling get_output() <- get_input() <- last-layer.get_output()
    You only need to call last layer's get_output(), then all layers's forward-pass will be called
"""

class Layer(object):
    """
    abstract class of layer
    """
    def __init__(self):
        self.params = []
        self.regs = []

    def get_output(self,train=False):
        raise NotImplementedError

    def get_input(self,train=False):
        """
        Key function to connect layers and compute forward-pass
        """
        if hasattr(self, 'previous'):
            return self.previous.get_output(train)
        else:
            return self.input

    def get_params(self):
        return self.params

    def get_regs(self):
        return self.regs

    def get_param_vals(self):
        """
        get layer parameter values (tensor -> numpy)
        """
        param_vals = []
        for p in self.params:
            param_vals.append(p.get_value())
        return param_vals
    
    def set_param_vals(self, param_vals):
        for (p, pval) in zip(self.params, param_vals):
            if p.eval().shape != pval.shape:
                raise Exception("[Error] in set_param_vals: input numpy params has different shape of model params")
            p.set_value(floatX(pval))

    def connect(self, layer):
        self.previous = layer


class AffineLayer(Layer):
    """
    Affine (fully connected) layer
    """
    def __init__(self, nb_input, nb_output, init='normal',
            activation='linear', reg_W=0.001, reg_b=0.,w_scale=1e-5):
        super(AffineLayer, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.nb_input = nb_input
        self.nb_output = nb_output
        
        # this symbolic variable will be used if this is the first layer
        self.input = T.matrix('input',dtype=theano.config.floatX)
        
        self.W = self.init((self.nb_input, self.nb_output),w_scale)
        self.b = shared_zeros((self.nb_output))
        self.reg_W = shared_scalar(reg_W)
        self.reg_b = shared_scalar(reg_b)
        self.params = [self.W, self.b]
        self.regs   = [self.reg_W, self.reg_b]

    
    # forward pass for the affine layer
    def get_output(self,train=False):
        X = self.get_input(train)
        return self.activation(T.dot(X,self.W) + self.b)

    # forward pass without activation
    # This is used to get score before softmax
    def get_output_score(self, train=False):
        X = self.get_input(train)
        return T.dot(X,self.W) + self.b


    def get_config(self):
        return {'name': self.__class__.__name__,
                'nb_input': self.nb_input,
                'nb_output': self.nb_output,
                'init': self.init.__name__,
                'activation': self.activation.__name__,
                'reg_W': self.reg_W}

class Dropout(Layer):
    """
    Dropout Layer
    Reference: http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    """
    def __init__(self, p, nb_input=None, uncertainty = False):
        """
        p: floatX, percentage of neurons want to drop, higher this value, more to drop
        uncertainty: Yarin Gal's Gaussian process uncertainty estimation

        Reference: http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html
        """
        super(Dropout, self).__init__()
        self.p = p
        self.uncertainty = uncertainty
        self.nb_input = nb_input
        #self.nb_output = nb_output

    def get_output(self, train=False):
        X = self.get_input(train)
        assert self.p >= 0
        retain_prob = 1. - self.p
        if train:
            X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX) / retain_prob
        elif not train and self.uncertainty:
            X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX) / retain_prob
        
        return X

    def get_config(self):
        return {"name":self.__class__.__name__,
                "p":self.p}

