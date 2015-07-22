import numpy as np
import sys
import theano
import theano.tensor as T
import time

sys.path.append('../../../deuNN/')

from deuNN import optimizers
from deuNN import losses
from deuNN import callbacks as cbks
from deuNN.layers import containers
from deuNN.models import NN

class SharpeNet(NN):
    def __init__(self,checkpoint_fn=None):
        self.layers = []
        self.params = []
        self.regs = []
        self.checkpoint_fn = checkpoint_fn
    
    def loss(self, z, y):
        y1 = T.max(z,axis=1)*y
        return -(T.mean(y1) / T.std(y1)) 
        #return -(T.mean(y1) / (T.std(y1)+T.std(T.max(z,axis=1)))) 
        #return -(T.mean(y1) / (T.std(y1)+1e-1*T.std(y)*T.std(T.max(z,axis=1)))) 

    def ir(self, z, y):
        y1 = T.max(z,axis=1)*y
        return (T.mean(y1) / T.std(y1))

    def _test(self,z, y):
        y1 = T.max(z,axis=1)*y
        return y1


    def compile(self, optimizer, reg_type='L2', learning_rate = 0.01,
            momentum=None, lr_decay=None,
            nesterov=False,rho=0.01, class_mode=None):
        """
        Build and compile theano graph functions
        Inputs:
            - optimizer: str, SGD method
            - loss: str, loss function method
        """
        self.learning_rate = learning_rate
        self.optimizer = optimizers.get(optimizer)
        self.optimizer.set_lr(learning_rate)
        #self.data_loss = losses.get(loss)
        self.reg_loss = losses.get(reg_type)
        self.class_mode = class_mode

        # NN input and output
        self.X_train    = self.get_input(train=True)
        self.X_test    = self.get_input(train=False)
        self.py_x_train = self.get_output(train=True)
        self.py_x_test = self.get_output(train=False)
        
        # output score instead of probs
        self.sy_x_test = self.get_output_score(train=False)

        #self.y    = T.zeros_like(self.py_x_train)
        self.y    = T.vector(dtype='float32')
        
        #data_loss_train = self.data_loss(self.py_x_train, self.y)
        data_loss_train = self.loss(self.py_x_train, self.y)
        reg_loss_train = self.reg_loss(self.params, self.regs)
        total_loss_train = data_loss_train + reg_loss_train
        
        #data_loss_test = self.data_loss(self.py_x_test, self.y)
        data_loss_test = self.loss(self.py_x_test, self.y)
        reg_loss_test = self.reg_loss(self.params, self.regs)
        total_loss_test = data_loss_test + reg_loss_test

        updates = self.optimizer.get_updates(total_loss_train, self.params)

        ins_train = [self.X_train, self.y]
        ins_test = [self.X_test, self.y]

        self._train = theano.function(
                inputs = ins_train,
                outputs = total_loss_train,
                updates = updates,
                allow_input_downcast=True)
        
        ir = self.ir(self.py_x_test,self.y)
        self._get_ir = theano.function(
                inputs = ins_test,
                outputs = ir,
                allow_input_downcast = True)

        self._get_pnl = theano.function(
                inputs = ins_test,
                outputs = self._test(self.py_x_test,self.y),
                allow_input_downcast = True)

        self._get_output = theano.function(
                inputs = [self.X_test],
                outputs = self.py_x_test,
                allow_input_downcast = True)
    
    def predict(self, test_X):
        return self._get_output(test_X)
    
    def test(self, test_X, test_y):
        test_ins = [test_X, test_y]
        test_acc = self._get_ir(*test_ins)
        print "*** Test Acc. = %.4f"%test_acc

    def get_pnl(self, test_X, test_y):
        test_ins = [test_X, test_y]
        return self._get_pnl(*test_ins)

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
        self.verbose = verbose
        (N, D) = train_X.shape
        print 'Training Data: %d x %d'%(N, D)
        print 'Validation Data: %d x %d'%(valid_X.shape[0], valid_X.shape[1])
        
        # mini-batch training
        print 'Start Training'
        start_time = time.clock()
        iter_num = 0
        best_valid_acc = -np.inf
        
        train_params = {'verbose':verbose,'nb_samples':N,'nb_epoch':nb_epoch}
        logger = cbks.baseLogger(train_params)
        history_log = cbks.History(train_params)
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
                train_loss = self._train(*train_ins)
                #train_acc = self._get_ir(*train_ins)
                train_acc = self._get_ir(train_X, train_y)
                batch_logs = {'loss':train_loss, 'size':batch_size}
                batch_logs['accuracy'] = train_acc
                logger.on_batch_end(iter_num, batch_logs)
                history_log.on_batch_end(iter_num, batch_logs)
                                      
            valid_acc = self._get_ir(*valid_ins)
            valid_loss = -1.*valid_acc
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
            epoch_logs = {'val_loss':valid_loss, 'val_acc':valid_acc}
            logger.on_epoch_end(epoch, epoch_logs)
            history_log.on_epoch_end(epoch,epoch_logs)
            if self.checkpoint_fn is not None:
                checkpoint.on_epoch_end(epoch,epoch_logs)
        history_log.on_train_end()
        end_time = time.clock()
        print "Training finished, best validation error %f"%(best_valid_acc)
        print 'The training run for %d epochs, with %f epochs/sec'%(nb_epoch,
                1.*nb_epoch / (end_time - start_time))
 
