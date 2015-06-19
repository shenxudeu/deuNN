import numpy as np

def one_hot(y, nb_classes=None):
    N = len(y)
    y = np.asarray(y, dtype='int32')
    if nb_classes is None:
        nb_classes = 1 + np.max(y)
    Y = np.zeros((N,nb_classes))
    Y[np.arange(N),y] = 1.
    return Y


