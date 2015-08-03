import numpy as np

def one_hot(y, nb_classes=None):
    N = len(y)
    y = np.asarray(y, dtype='int32')
    if len(y.shape) > 1:
        y = y.flatten()
    if nb_classes is None:
        nb_classes = 1 + np.max(y)
    Y = np.zeros((N,nb_classes))
    Y[np.arange(N),y] = 1.
    return Y

def np_softmax(X):
    """
    numpy version of softmax
    X: np.array, nb_samples * nb_features 
    """
    e_x = np.exp(X - X.max(axis=1,keepdims=True))
    out = e_x / e_x.sum(axis=1, keepdims=True)
    return out

