"""
Backback class: print out and save out training process
"""

import theano
import theano.tensor as T
import warnings
import time
import numpy as np
import logging
import pprint
import os

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
    def __init__(self, params):
        
        self._set_params(params)
        self.verbose = self.params['verbose']
    #def on_train_begin(self, params):
    #    self._set_params(params)
    #    self.verbose = self.params['verbose']

    def on_epoch_begin(self, epoch):
        if self.verbose:
            if self.verbose == True:
                print "Epoch %d "%epoch
            self.progbar = Progbar(target=self.params['nb_samples'],
                    verbose=self.verbose)
            self.current,self.tot_loss, self.tot_acc = 0, 0., 0.

    def on_batch_begin(self, batch):
        if not self.verbose:
            return
        if self.current < self.params['nb_samples']:
            self.log_values = []
    
    def on_batch_end(self, batch,logs={}):
        if not self.verbose:
            return
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
        if not self.verbose:
            return
        self.log_values = []
        self.log_values.append(('loss', self.tot_loss / self.current))
        self.log_values.append(('acc.', self.tot_acc / self.current))
        self.log_values.append(('val_loss', logs.get('val_loss')/1.))
        self.log_values.append(('val_acc.', logs.get('val_acc')/1.))
        self.progbar.update(self.params['nb_samples'], self.log_values)


class History(CallBack):
    def __init__(self, params,hist_fn, config):
        self._set_params(params)
        self.verbose = self.params['verbose']
        if os.path.isfile(hist_fn):
            os.remove(hist_fn)
        logging.basicConfig(filename=hist_fn,level=logging.INFO,format='%(asctime)s %(message)s')
        config_str = pprint.pformat(config, indent = 4)
        logging.info(config_str)


    def on_train_begin(self):
        self.epoch = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.log_values = []
        #if not self.verbose: # not verbose means no base loger
        #    self.progbar = Progbar(target=self.params['nb_epoch'],
        #            verbose=True)
        #    self.current = 0

    def on_epoch_begin(self, epoch):
        self.tot_loss = 0.
        self.tot_acc = 0.

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size',0)
        #self.current += batch_size
        self.tot_loss += logs.get('loss') * batch_size
        self.tot_acc += logs.get('accuracy') * batch_size

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        self.loss.append(self.tot_loss / self.params['nb_samples'])
        self.acc.append(self.tot_acc / self.params['nb_samples'])
        #self.loss.append(logs.get('loss'))
        #self.acc.append(logs.get('acc.'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        log_str = 'On epoch %d, train_loss = %.6f, train_acc = %.6f'%(epoch, self.loss[-1],self.acc[-1])
        log_str += ', val_loss = %.6f, val_acc = %.6f'%(self.val_loss[-1],self.val_acc[-1])
        logging.info(log_str)
        #if self.verbose:
        #    return
        #self.current += 1
        #if self.current < self.params['nb_epoch']:
        #    self.log_values = []
        #    self.log_values.append(('loss',self.loss[-1]))
        #    self.log_values.append(('acc.',self.acc[-1]))
        #    self.log_values.append(('val_loss',self.val_loss[-1]))
        #    self.log_values.append(('val_acc',self.val_acc[-1]))
        #    self.progbar.update(self.current,self.log_values)
    
    def on_train_end(self):
        logging.info('Training END')
        #hist_f.close()
        #np.savetxt('.train_log.csv',(
        #    self.epoch,self.loss, self.acc,self.val_loss,self.val_acc),
        #    delimiter=',')
        

class ModelCheckPoint(CallBack):
    def __init__(self, fname, model):
        super(ModelCheckPoint, self).__init__()
        self.model = model
        self.fname = fname
        self.loss = []
        self.best_loss = np.Inf
        self.best_val_acc = 0.
        
    def on_epoch_end(self, epoch, logs={}, verbose=True):
        #if self.best_val_acc < logs.get('val_acc') and epoch > 5:
        if logs.get('val_loss') < self.best_loss and epoch > 5:
            if verbose:
                print "On epoch %d: validation loss improved from %0.5f to %0.5f, saving model to %s"%(epoch, self.best_loss, logs.get('val_loss')*1., self.fname)
            self.best_val_acc = logs.get('val_acc')*1.
            self.best_loss = logs.get('val_loss')*1.
            self.model.save_model(self.fname, overwrite=True)

