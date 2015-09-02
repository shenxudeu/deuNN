import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from .. import activations, initializations
from .. utils.theano_utils import shared_zeros, shared_scalar
from .. layers.core import Layer

class Flatten(Layer):
    """
    Flatten a multi-dim volumn to 2D
    Assume the first dim to be nb_samples.
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def get_output(self, train=False):
        X = self.get_input(train)
        return T.flatten(X, outdim=2)

    def get_config(self):
        return {'name':self.__class__.__name__}

class Convolution2D(Layer):
    """
    2D Convolution Layer
    Parameters:
     - nb_filter: number of filters K, the depth of 3D conv volume
     - stack_size: last layer's depth D_1
     - nb_row: Filter width F_row
     - nb_col: Filter height F_col
     - subsample: Stride S
    
    Example of 2D Conv Layer input->output dim calculation
    Input: a volume of size W_1 x H_1 x D_1
    Hyperparameters of Conv Layer:
        nb_filter = K
        nb_row, nb_col = F, F
        the stride, subsample = S, S
        the amount of padding P
    
    Output: a volume of size W_2 x H_2 x D2
    W_2 = (W_1 - F + 2P)/S + 1
    H_2 = (H_1 - F + 2P)/S + 1
    D_2 = K

    Note: W_2 and H_2 must be integer
    Reference: http://cs231n.github.io/convolutional-networks/
    """
    def __init__(self, nb_filter, stack_size, nb_row, nb_col,
            init='glorot_uniform',activation='linear',
            border_mode='valid',subsample=(1,1),
            reg_W=None, reg_b=0.,
            w_scale=1e-5):

        if border_mode not in {'valid', 'full'}:
            raise Exception("Invalid boder mode for ConvNet 2D")

        super(Convolution2D,self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        
        self.nb_filter = nb_filter
        self.stack_size = stack_size
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.subsample = subsample
    
        self.border_mode = border_mode

        self.input = T.tensor4()
        self.W_shape = (nb_filter, stack_size, nb_row, nb_col)
        self.W = self.init(self.W_shape, w_scale)
        self.b = shared_zeros((nb_filter,))
        self.reg_W = shared_scalar(reg_W)
        self.reg_b = shared_scalar(reg_b)
        self.params = [self.W, self.b]
        self.regs = [self.reg_W, self.reg_b]


    # forward pass for 2D conv layer
    def get_output(self,train=False):
        X = self.get_input(train)
        conv_out = T.nnet.conv2d(X, self.W,
                border_mode=self.border_mode, subsample=self.subsample)
        
        return self.activation(conv_out + self.b.dimshuffle('x',0,'x','x'))

        
    def get_config(self):
        return {'name': self.__class__.__name__,
                'nb_filter': self.nb_filter,
                'stack_size':self.stack_size,
                'nb_row':self.nb_row,
                'nb_col':self.nb_col,
                'init':self.init.__name__,
                'activation':self.activation.__name__,
                'border_mode':self.border_mode,
                'subsample':self.subsample,
                'reg_W':self.reg_W
                }


class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2,2), stride=None, ignore_border=False):
        super(MaxPooling2D,self).__init__()
        self.input = T.tensor4()
        self.poolsize = pool_size
        self.stride = stride
        self.ignore_border = ignore_border

    def get_output(self,train):
        X = self.get_input(train)
        output = downsample.max_pool_2d(X, ds=self.poolsize,
                ignore_border = self.ignore_border)
        return output

    def get_config(self):
        return {'name':self.__class__.__name__,
                'poolsize':self.poolsize,
                'ignore_border':self.ignore_border,
                'stride':self.stride}


class ZeroPadding2D(Layer):
    def __init__(self,P = 1):
        super(ZeroPadding2D,self).__init__()
        self.P = P
        self.input = T.tensor4()

    def get_output(self,train):
        X = self.get_input(train)
        P = self.P
        in_shape = X.shape
        out_shape = (in_shape[0],in_shape[1],
                in_shape[2] + 2 * P,
                in_shape[3] + 2 * P)
        out = T.zeros(out_shape)
        indices = (slice(None),
                   slice(None),
                   slice(P,in_shape[2]+P),
                   slice(P,in_shape[3]+P))
        return T.set_subtensor(out[indices], X)

    def get_config(self):
        return {'name':self.__class__.__name__,
                'P':self.P}


class Convolution1D(Layer):
    """
    1D Conv Layer: It is implemented using T.nnet.conv2D function
    It is also called time-delayed Layer
    """
    def __init__(self, nb_filter, stack_length, filter_length,
            init='glorot_uniform',activation='linear',
            border_mode='valid',subsample_length=1,
            reg_W=None, reg_b=0., w_scale=1e-5):
        
        if border_mode not in {'valid','full'}:
            raise Exception("Invalid border mode for ConvNet 1D")

        super(Convolution1D,self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.nb_filter = nb_filter
        self.stack_length = self.stack_length
        self.filter_length = filter_length
        self.subsample = (1,subsample_length)

        self.border_mode = border_mode

        self.input = T.tensor3()
        self.W_shape = (nb_filter, stack_length, filter_length, 1)
        self.W = self.init(self.W_shape, w_scale)
        self.b = shared_zeros((nb_filter,))
        self.reg_W = shared_scalar(reg_W)
        self.reg_b = shared_scalar(reg_b)
        self.params = [self.W, self.b]
        self.regs = [self.reg_W, self.reg_b]

    def get_output(self,train=False):
        X = self.get_input(train)
        X = T.reshape(X, (X.shape[0],X.shape[1],X.shape[2],1)).dimshuffle(0,2,1,3)
        
        conv_out = T.nnet.conv2d(X, self.W,
                border_mode=self.border_mode, subsample=subsample)
        
        output = self.activation(conv_out, self.b.dimshuffle('x',0,'x','x'))
        
        return T.reshape(output, (
            output.shape[0],output.shape[1],output.shape[2])).dimshuffle(0,2,1)

    def get_config(self):
        return {'name':self.__class__.__name__,
                'nb_filter':self.nb_filter,
                'stack_length':self.stack_length,
                'filter_length':self.filter_length,
                'init':self.init.__name__,
                'activation':self.activation.__name__,
                'border_mode':self.border_mode,
                'subsample':self.subsample,
                'reg_W':self.reg_W}


class MaxPooling1D(Layer):
    def __init__(self, pool_length=2, stride=None, ignore_border=True):
        super(MaxPooling1D,self).__init__()
        self.input = T.tensor3()
        self.pool_size = (1,pool_length)
        if stride:
            self.stride = (1,stride)
        else:
            self.stride = stride

        self.ignore_border = ignore_border


    def get_output(self, train=False):
        X = self.get_input(train)
        X = T.reshape(X, (X.shape[0],X.shape[1],X.shape[2],1)).dimshuffle(0,1,3,2)
        output = downsample.max_pool_2d(X, ds=self.pool_size, st=self.stride,
                ignore_border=self.ignore_border)
        output = output.dimshuffle(0,1,3,2)
        return T.reshape(output,(output.shape[0],output.shape[1],output.shape[2]))

    def get_config(self):
        return {'name':self.__class__.__name__,
                'stride':self.stride,
                'pool_size':self.pool_size,
                'ignore_border':self.ignore_border}





