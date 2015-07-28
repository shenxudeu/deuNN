import numpy as np
import sys, os
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import time

sys.path.append('../../deuNN/')

from deuNN.datasets import cifar_10

import pdb

[train_X, train_y, valid_X, valid_y, test_X, test_y] = cifar_10.load_data()
