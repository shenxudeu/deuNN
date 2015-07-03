import theano
import theano.tensor as T
import numpy as np
import time

from . import optimizers
from . import losses
from . import callbacks as cbks
from .layers import containers

import pdb

class NN(containers.Sequential):
    """
    Outside Class NN: derived from sequential container
    compile theano graph funtions
    get updates and SGD, fit model with input data
    """
    def __init__(self):
        self.layers = []
        self.params = []
        self.regs = []

    def get_config(self):
        configs = {}
        for (i,l) in enumerate(self.layers):
            configs['layer-i'] = l.get_config()
        return configs

    def compile(self, optimizer, loss, reg_type='L2', learning_rate = 0.01,
            class_mode="categorical"):
        """
        Build and compile theano graph functions
        Inputs:
            - optimizer: str, SGD method
            - loss: str, loss function method
        """
        self.optimizer = optimizers.get(optimizer)
        self.optimizer.set_lr(learning_rate)
        self.data_loss = losses.get(loss)
        self.reg_loss = losses.get(reg_type)

        # NN input and output
        self.X    = self.get_input()
        self.py_x = self.get_output()
        self.y    = T.zeros_like(self.py_x)

        data_loss = self.data_loss(self.py_x, self.y)
        reg_loss = self.reg_loss(self.params, self.regs)
        total_loss = data_loss + reg_loss

        if class_mode == 'categorical':
            accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1),
                                   T.argmax(self.py_x, axis=-1)))
        
        self.class_mode = class_mode

        updates = self.optimizer.get_updates(data_loss, self.params)

        ins = [self.X, self.y]

        self._train = theano.function(
                inputs = ins,
                outputs = total_loss,
                updates = updates,
                allow_input_downcast=True)
        self._train_acc = theano.function(
                inputs = ins,
                outputs = [total_loss, accuracy],
                updates = updates,
                allow_input_downcast=True)
        self._get_acc_loss = theano.function(
                inputs = ins,
                outputs = [total_loss, accuracy],
                allow_input_downcast = True)
        self._test = theano.function(
                inputs = ins,
                outputs = accuracy,
                allow_input_downcast=True)

    def train(self, X, y , accuracy=False):
        ins = [X, y]
        if accuracy:
            return self._train_acc(*ins)
        else:
            return self._train(*ins)

    def test(self, X, y):
        ins = [X,y]
        return self._test(*ins)

    def fit(self, train_X, train_y, valid_X, valid_y,
            batch_size=50, nb_epoch=20, verbose=True):
        """
        NN fit function
        Inputs:
            - train_X: np.array
            - train_y: np.array, one-hot if classification problem
            - valid_X: np.array
            - valid_y: np.array, one-hot if classification problem
            - batch_size: int
            - nb_epoch: int, number of epoch
            - verbose: bool
        """
        (N, D) = train_X.shape
        print 'Training Data: %d x %d'%(N, D)
        print 'Validation Data: %d x %d'%(valid_X.shape[0], valid_X.shape[1])
        
        # mini-batch training
        print 'Start Training'
        start_time = time.clock()
        iter_num = 0
        #acc_frequency = 200
        best_valid_acc = -np.inf
        
        train_params = {'verbose':verbose,'nb_samples':N}
        logger = cbks.baseLogger(train_params)
        #logger.on_train_begin(train_params)
        
        valid_ins = [valid_X, valid_y]
        for epoch in xrange(nb_epoch):
            logger.on_epoch_begin(epoch)
            for start, end in zip(range(0,N,batch_size), range(batch_size,N,batch_size)):
                iter_num += 1
                logger.on_batch_begin(iter_num)
                train_ins = [train_X[start:end], train_y[start:end]]
                #valid_ins = [valid_X, valid_y]
                #train_loss = self._train(*train_ins)
                if verbose:
                    [train_loss, train_acc] = self._train_acc(*train_ins)
                else:
                    train_loss = self._train(*train_ins)
                    train_acc = np.nan
                batch_logs = {'loss':train_loss, 'size':batch_size}
                batch_logs['accuracy'] = train_acc
                logger.on_batch_end(iter_num, batch_logs)
                                      
                #if iter_num % acc_frequency == 0:
                #    [train_loss, train_acc] = self._train_acc(*train_ins)
                #    valid_acc = self._test(*valid_ins)
                #    if valid_acc > best_valid_acc:
                #        best_valid_acc = valid_acc
                #    #print 'Iteration %d, finish epoch %d / %d: cost %f, train: %f, val: %f'%(iter_num, epoch, nb_epoch, train_loss, train_acc, valid_acc)
                #else:
                #    train_loss = self._train(*train_ins)
                #    train_acc = 0.
                #    if 1:
                #        print '\nIteration %d: cost %f\n'%(iter_num,train_loss)
                
                batch_logs = {'loss':train_loss, 'size':batch_size}
                batch_logs['accuracy'] = train_acc
                logger.on_batch_end(iter_num, batch_logs)
            [valid_loss, valid_acc] = self._get_acc_loss(*valid_ins)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
            epoch_logs = {'val_loss':valid_loss, 'val_acc':valid_acc}
            logger.on_epoch_end(epoch, epoch_logs)
        end_time = time.clock()
        print "Training finished, best validation error %f"%(best_valid_acc)
        print 'The training run for %d epochs, with %f epochs/sec'%(nb_epoch,
                1.*nb_epoch / (end_time - start_time))
    

    def get_test_accuracy(self, X, y):
        test_acc = self.test(X, y)
        print 'Testing Accuracy %f'%test_acc

    def save_model(self,filepath, overwrite=False):
        import h5py
        import os.path
        if not overwrite and os.path.isfile(filepath):
            import sys
            get_input = input
            if sys.version_info[:2] <= (2,7):
                get_input = raw_input
            overwrite = get_input('[Warning] %s already exists - overwrite? [y/n]'%filepath)
            while overwrite not in ['y','n']:
                overwrite = get_input('Please enter [y] yes or [n] no')
            if overwrite == 'n':
                return
            print '[Tip] You can set overwrite=True next time'

        f = h5py.File(filepath,'w')
        f.attrs['nb_layers'] = len(self.layers)
        for k, l in enumerate(self.layers):
            g = f.create_group('layer_{}'.format(k))
            weights = l.get_param_vals()
            f.attrs['nb_params'] = len(weights)
            for n, param in enumerate(weights):
                param_name = 'param_{}'.format(n)
                param_dset = g.create_dataset(param_name, param.shape, dtype=param.dtype)
                param_dset[:] = param
        f.flush()
        f.close()
        print 'Model saved as %s successfully!'%filepath
    
    def load_model(self,filepath):
        import h5py
        f = h5py.File(filepath,'r')
        for k in xrange(f.attrs['nb_layers']):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(i)] for i in xrange(g.attrs['nb_params'])]
            self.layers[k].set_param_vals(weights)
        f.close()

