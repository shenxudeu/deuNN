import numpy as np
import os, sys
import gzip
import cPickle

def load_data(data_name='mnist.pkl.gz'):
    """
    Load MNIST Dataset
    Inputs:
        - data_name: str, the filename of saved data
    Outputs:
        - rval: list of tuples, each elements is a numpy array, in order of:
                train_set, valid_set, test_set,
                xx_set: (xx_X, xx_y)
    """
    data_dir, data_file = os.path.split(dataset)
    
    # download MNIST if not found
    if not os.path.isfile(dataset):
        import urllib
        origin = (
                'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading MNIST from %s' % origin
        assert urllib.urlretrieve(origin, dataset)
            
    print "Loading Data ..."

    with gzip.open(dataset, 'rb') as handle:
        train_set, valid_set, test_set = cPickle.load(handle)
    
    return [train_set, valid_set, test_set]
