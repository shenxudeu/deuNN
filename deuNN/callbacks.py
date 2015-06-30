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

class baseLogger(CallBack):
    def on_train_begin(self, params):
        self._set_params(params)
        self.verbose = self.params['verbose']

    def on_epoch_begin(self, epoch):
        if self.verbose:
            print "Epoch %d "%epoch
            self.progbar = Progbar(target=self.params['nb_samples'],
                    verbose=self.verbose)
        self.current,self.tot_loss, self.tot_acc = 0, 0., 0.

    def on_batch_begin(self, batch):
        if self.current < self.params['nb_samples']:
            self.log_values = []
    
    def on_batch_end(self, batch,logs={}):
        batch_size = logs.get('size', 0)
        self.current += batch_size

        loss = logs.get('loss')
        self.log_values.append(('loss',loss))
        self.tot_loss += loss * batch_size
        acc = logs.get('accuracy')
        self.log_values.append(('acc.',acc))
        self.tot_acc += acc * batch_size

        if self.current < self.params['nb_samples']:
            self.progbar.update(self.current, self.log_values)

    def on_epoch_end(self, epoch, logs={}):
        self.log_values.append(('loss', self.tot_loss / self.current))
        self.log_values.append(('acc.', self.tot_acc / self.current))
        self.log_values.append(('val_loss', logs.get('val_loss')))
        self.log_values.append(('val_acc.', logs.get('val_acc')))
        self.progbar.update(self.current, self.log_values)


class History(CallBack):
    def on_train_begin(self):
        self.epoch = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def on_epoch_begin(self, epoch):
        self.current = 0
        self.tot_loss = 0.
        self.tot_acc = 0.

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size',0)
        self.current += batch_size

        self.tot_loss += logs.get('loss') * batch_size
        self.tot_acc += logs.get('acc') * batch_size

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        self.loss.append(self.tot_loss / self.current)
        self.acc.append(self.tot_acc / self.current)
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

class ModelCheckPoint(CallBack):
    def __init__(self, fname, model):
        super(Callback, self).__init__()
        self.model = model
        self.fname = fname
        self.loss = []
        self.best_loss = np.Inf
        self.best_val_acc = 0.
        
    def on_epoch_end(self, epoch, logs={}):
        if self.best_val_acc < logs.get('val_acc'):
            print "On epoch %d: validation loss improved from %0.5f to %0.5f, saving model to %s"%(self.best_val_acc, logs.get('val_acc'), self.fname)
            self.best_val_acc = logs.get('val_acc')
            self.model.save_weights(self.fname, overwrite=True)

