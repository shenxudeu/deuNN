import numpy as np
import os, sys
import gzip
import cPickle

def load_cifar_batch(filename):
    """ load single batch of cifar-10 """
    with open(filename, 'rb') as f:
        datadict = cPickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype('float')
        Y = np.array(Y)
        return X, Y

def load_cifar10(ROOT):
    """ load all of cifar-10 """
    xs,ys = [], []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_cifar_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def load_data(num_training=49000, num_validation=1000, num_test=1000):
    cifar10_dir = "cifar-10-batches-py"

    if not os.path.exists(cifar10_dir):
        import urllib
        origin = (
                'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        )
        print "Downloading CIFAR-10 from %s"% origin
        assert urllib.urlretrieve(origin,"cifar-10-python.tar.gz")

    print "Loading Data ..."

    import tarfile
    with tarfile.open("cifar-10-python.tar.gz",'r') as t:
        f = t.extractall()

    X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    return X_train, y_train, X_val, y_val, X_test, y_test

