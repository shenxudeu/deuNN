import theano
import theano.tensor as T
import numpy as np
import time

from . import SGD
from . import losses
from .layers import containers

class NN(containers.Sequential):
    """
    Outside Class NN: derived from sequential container
    compile theano graph funtions
    get updates and SGD, fit model with input data
    """
    def __init__(self):
        self.layers = []
        self.params = []

    def get_config(self):
        configs = {}
        for (i,l) in enumerate(self.layers):
            configs['layer-i'] = l.get_config()
        return configs

    def compile(self, optimizer, loss, learning_rate = 0.01,
            class_mode="categorical"):
        """
        Build and compile theano graph functions
        Inputs:
            - optimizer: str, SGD method
            - loss: str, loss function method
        """
        self.optimizer = SGD.get(optimizer)
        self.optimizer.set_lr(learning_rate)
        self.data_loss = losses.get(loss)

        # NN input and output
        self.X    = self.get_input()
        self.py_x = self.get_output()
        self.y    = T.zeros_like(self.py_x)

        data_loss = self.data_loss(self.py_x, self.y)

        if class_mode == 'categorical':
            accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1),
                                   T.argmax(self.py_x, axis=-1)))
        
        self.class_mode = class_mode

        updates = self.optimizer.get_updates(data_loss, self.params)

        ins = [self.X, self.y]

        self._train = theano.function(
                inputs = ins,
                outputs = data_loss,
                updates = updates,
                allow_input_downcast=True)
        self._train_acc = theano.function(
                inputs = ins,
                outputs = [data_loss, accuracy],
                updates = updates,
                allow_input_downcast=True)
        self._test = theano.funcion(
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
        acc_frequency = 200
        best_valid_acc = -np.inf
        for epoch in xrange(nb_epoch):
            for start, end in zip(range(0,N,batch_size), range(batch_size,N,batch_size)):
                iter_num += 1
                train_ins = [train_X[start:end,:], train_y[start:end,:]]
                valid_ins = [valid_X, valid_y]
                train_loss = self._train(train_ins)

                if iter_num % acc_frequency == 0:
                    [train_loss, train_acc] = self._train_acc(*train_ins)
                    valid_acc = self._test(*valid_ins)
                    print 'Iteration %d, finish epoch %d / %d: cost %f, train: %f, val: %f'%(iter_num, epoch, nb_epoch, train_loss, train_acc, valid_acc)
                else:
                    train_loss = self._train(train_ins)
                    if verbose:
                        print 'Iteration %d: cost %f'%train_loss

        end_time = time.clock()
        print 'The training run for %d epochs, with %f epochs/sec'%(nb_epoch,
                1.*nb_epoch / (end_time - start_time))





