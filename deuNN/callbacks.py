"""
Backback class: print out and save out training process
"""

import theano
import theano.tensor as T
import warnings
import time
import numpy as np

from .utils.generic_utils import Progbar

import pdb


class CallBack(object):
    def __init__(self):
        pass
    def _set_params(self, params):
        self.params = params
    def _set_model(self, model):
        self.model = model
    def on_train_begin(self):
        pass
    def on_train_end(self):
        pass
    def on_epoch_begin(self):
        pass
    def on_epoch_end(self):
        pass
    def on_batch_begin(self):
        pass
    def on_batch_end(self):
        pass

