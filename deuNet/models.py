import theano
import theano.tensor as T
import numpy as np
import time

from . import optimizers
from . import losses
from . import callbacks as cbks
from .layers import containers
from optimizers import SGD, RMSprop

import pdb

class NN(containers.Sequential):
    """
    Outside Class NN: derived from sequential container
    compile theano graph funtions
    get updates and SGD, fit model with input data
    """
    def __init__(self, checkpoint_fn=None,log_fn='.default_log.log'):
        self.layers = []
        self.params = []
        self.regs = []
        self.checkpoint_fn = checkpoint_fn
        self.log_fn = log_fn

    def get_config(self):
        configs = {}
        for (i,l) in enumerate(self.layers):
            configs['layer-%d'%i] = l.get_config()
        return configs

    def get_lr(self):
        return self.learning_rate
    
    def get_w_decay(self):
        config = self.get_config()
        w_decay = 0.
        for (i,l) in enumerate(self.layers):
            if 'reg_W' in config['layer-%d'%i]:
                wd = config['layer-%d'%i]['reg_W'].get_value()/1.
                if w_decay < wd:
                    w_decay = wd
        return w_decay + 1e-12

    def compile(self, optimizer, loss, reg_type='L2', learning_rate = 0.01,
            class_mode="categorical",momentum=None,lr_decay=None,
            nesterov=False,rho=0.01):
        """
        Build and compile theano graph functions
        Inputs:
            - optimizer: str, SGD method
            - loss: str, loss function method
        """
        self.learning_rate = learning_rate
        self.optimizer = optimizers.get(optimizer)
        self.optimizer.set_lr(learning_rate)
        if optimizer == 'SGD':
            self.optimizer.set_momentum(momentum)
            self.optimizer.set_lr_decay(lr_decay)
            self.optimizer.set_nesterov(nesterov, momentum)
        elif optimizer == 'RMSprop' or optimizer == 'Adadelta':
            self.optimizer.set_rho(rho)
        #self.optimizer = RMSprop(lr=1e-2,rho=0.9)

        self.data_loss = losses.get(loss)
        self.reg_loss = losses.get(reg_type)

        # NN input and output
        self.X_train    = self.get_input(train=True)
        self.X_test    = self.get_input(train=False)
        self.py_x_train = self.get_output(train=True)
        self.py_x_test = self.get_output(train=False)
        
        # output score instead of probs
        self.sy_x_test = self.get_output_score(train=False)
        
        #if class_mode == "categorical":
        self.y    = T.zeros_like(self.py_x_train)


        data_loss_train = self.data_loss(self.py_x_train, self.y)
        reg_loss_train = self.reg_loss(self.params, self.regs)
        #total_loss_train = data_loss_train
        total_loss_train = data_loss_train + reg_loss_train
        
        data_loss_test = self.data_loss(self.py_x_test, self.y)
        reg_loss_test = self.reg_loss(self.params, self.regs)
        total_loss_test = data_loss_test + reg_loss_test
        #total_loss_test = data_loss_test

        if class_mode == 'categorical':
            accuracy_train = T.mean(T.eq(T.argmax(self.y, axis=-1),
                                   T.argmax(self.py_x_train, axis=-1)))
            accuracy_test = T.mean(T.eq(T.argmax(self.y, axis=-1),
                                   T.argmax(self.py_x_test, axis=-1)))
        elif class_mode == 'regression':
            acc_func = losses.get('mean_absolute_error')
            accuracy_train = acc_func(self.py_x_train,self.y)
            accuracy_test = acc_func(self.py_x_test, self.y)
        
        self.class_mode = class_mode

        updates = self.optimizer.get_updates(total_loss_train, self.params)

        ins_train = [self.X_train, self.y]
        ins_test = [self.X_test, self.y]

        self._train = theano.function(
                inputs = ins_train,
                outputs = total_loss_train,
                updates = updates,
                allow_input_downcast=True)
        self._train_acc = theano.function(
                inputs = ins_train,
                outputs = [total_loss_train, accuracy_train],
                updates = updates,
                allow_input_downcast=True)
        self._get_acc_loss = theano.function(
                inputs = ins_test,
                outputs = [total_loss_test, accuracy_test],
                allow_input_downcast = True)
        self._test = theano.function(
                inputs = ins_test,
                outputs = accuracy_test,
                allow_input_downcast=True)

        self._test_score = theano.function(
                inputs = [self.X_test],
                outputs = self.sy_x_test,
                allow_input_downcast=True)

        self._get_output = theano.function(
                inputs = [self.X_test],
                outputs = self.py_x_test,
                allow_input_downcast=True)

        #grads_data = self.optimizer.get_gradients(total_loss_train, self.params)
        ##grads_data = self.optimizer.get_gradients(data_loss_train, self.params)
        #self._get_grads = theano.function(
        #        inputs = ins_train,
        #        outputs = grads_data,
        #        allow_input_downcast=True)

        #self._get_prob = theano.function(
        #        inputs = [self.X_train],
        #        outputs = self.py_x_train,
        #        allow_input_downcast=True)
        #layer1_output = self.layers[0].get_output()
        #self._get_layer1_out = theano.function(
        #        inputs = [self.X_train],
        #        outputs = layer1_output,
        #        allow_input_downcast=True)

    def train(self, X, y , accuracy=False):
        ins = [X, y]
        if accuracy:
            return self._train_acc(*ins)
        else:
            return self._train(*ins)

    def test(self, X, y):
        ins = [X,y]
        return self._test(*ins)

    def predict(self, X):
        return self._get_output(X)

    def predict_uncertainty(self, X, nb_resample,beforeActivation=True):
        """
        Implementation of Yarin Gal's Bayesian Neural network outputs
        http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html
        """
        from .utils.np_utils import np_softmax
        probs = []
        for _ in xrange(nb_resample):
            if beforeActivation:
                probs += [self._test_score(X)]
            else:
                probs += [self._get_output(X)]
        predictive_mean = np.mean(probs, axis=0)
        predictive_variance = np.var(probs, axis=0)
        tau = self.get_lr() / self.get_w_decay()
        predictive_variance += tau**-1
        predictive_std = np.sqrt(predictive_variance)
        
        if self.class_mode == "categorical":
            # hard code softmax here, assume softmax conversion for all classifications
            pred_prob_mean = np_softmax(predictive_mean)
            pred_prob_ps1std = np_softmax(predictive_mean+predictive_std)
            pred_prob_ms1std = np_softmax(predictive_mean-predictive_std)
            return predictive_mean, predictive_std, pred_prob_mean
        
        return predictive_mean, predictive_std

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
        self.verbose=verbose
        
        N = train_X.shape[0]
        N_val = valid_X.shape[0]
        print 'Training Data: ', train_X.shape
        print 'Validation Data: ', valid_X.shape
        
        # mini-batch training
        print 'Start Training'
        start_time = time.clock()
        iter_num = 0
        best_valid_acc = -np.inf
        best_valid_loss = np.inf
        
        train_params = {'verbose':verbose,'nb_samples':N,'nb_epoch':nb_epoch}
        logger = cbks.baseLogger(train_params)
        history_log = cbks.History(train_params, self.log_fn)
        if self.checkpoint_fn is not None:
            checkpoint = cbks.ModelCheckPoint(self.checkpoint_fn,self)
        history_log.on_train_begin()
        
        valid_ins = [valid_X, valid_y]
        for epoch in xrange(nb_epoch):
            logger.on_epoch_begin(epoch)
            history_log.on_epoch_begin(epoch)
            for start, end in zip(range(0,N,batch_size), range(batch_size,N,batch_size)):
                iter_num += 1
                logger.on_batch_begin(iter_num)
                train_ins = [train_X[start:end], train_y[start:end]]
                [train_loss, train_acc] = self._train_acc(*train_ins)
                #grads_val = self._get_grads(*train_ins)
                #prob_val = self._get_prob(train_X[start:end])
                #test_val = self._get_layer1_out(train_X[start:end])
                #pdb.set_trace()
                batch_logs = {'loss':train_loss, 'size':batch_size}
                batch_logs['accuracy'] = train_acc
                logger.on_batch_end(iter_num, batch_logs)
                history_log.on_batch_end(iter_num, batch_logs)
                                      
            #[valid_loss, valid_acc] = self._get_acc_loss(*valid_ins)
            # batch forward on valid dataset, solve the out of memory issue
            valid_loss_total, valid_acc_total = 0., 0.
            nb_seen = 0.
            for start, end in zip(range(0,N_val,batch_size), range(batch_size,N_val,batch_size)):
                [val_loss_, val_acc_] = self._get_acc_loss(valid_X[start:end],valid_y[start:end])
                valid_loss_total += val_loss_ * batch_size
                valid_acc_total += val_acc_ * batch_size
                nb_seen += batch_size
            valid_loss = valid_loss_total / nb_seen
            valid_acc  = valid_acc_total  / nb_seen
            
            #if valid_acc > best_valid_acc:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_acc = valid_acc
            epoch_logs = {'val_loss':valid_loss, 'val_acc':valid_acc}
            logger.on_epoch_end(epoch, epoch_logs)
            history_log.on_epoch_end(epoch,epoch_logs)
            if self.checkpoint_fn is not None:
                checkpoint.on_epoch_end(epoch,epoch_logs,verbose=verbose)
        history_log.on_train_end()
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
            g.attrs['nb_params'] = len(weights)
            for n, param in enumerate(weights):
                param_name = 'param_{}'.format(n)
                param_dset = g.create_dataset(param_name, param.shape, dtype=param.dtype)
                param_dset[:] = param
        f.flush()
        f.close()
        if self.verbose:
            print 'Model saved as %s successfully!'%filepath
    
    def load_model(self,filepath):
        import h5py
        f = h5py.File(filepath,'r')
        for k in xrange(f.attrs['nb_layers']):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(i)] for i in xrange(g.attrs['nb_params'])]
            self.layers[k].set_param_vals(weights)
        f.close()

