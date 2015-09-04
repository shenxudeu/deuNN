"""
word2vec.py: word2vec models

References:
"Efficient Estimation of Word Representations in Vector Space", 
http://arxiv.org/pdf/1301.3781v3.pdf
http://cs224d.stanford.edu/lecture_notes/LectureNotes1.pdf
"""

import theano
import theano.tensor as T

from .. import activations, initializations
from .. layers.core import Layer
from .. utils.theano_utils import sharedX


class WordContext(Layer):
    """
    This is a very neat implementation of skip-grams model.
    
    Understanding 1:
    Skip-grams model is MLP with 1 hidden layer without any
    activations. It inputs a center word and predict its 
    surrounding words. softmax and cross-entropy is applied.
    |v|: vocab size
    n: word representation size
    C: surrounding window size
    X: 1 x |v|, input one word with one-hot
    Ww: |v| x n, inner world embedding matrix
    Wc: |v| x n, outer world embedding matrix
    h: 1 x n   = dot(X, Ww)
    y: 1 x |v| = dot(h, Wc.T)
    target can be think as |v| x 1 with surrounding words = 1
    softmax and cross-entropy can be applied as lost function.
    The training process is normal back-prop with SGD.

    Understanding 2:
    As described in [1] sector 2, we can think the dot product with one-hot as "Column Selection".
    Then we can treat the NN (1 -> C) as multiple 1->1 simple weight products.
    Then we can orginize the training data in one sentense (sequence) as mulitple samples with 1 input and 1 output.
    We do the column selection Ww -> W: 1 x n
                               Wc -> C: 1 x n
    the matrix product of two vectors comes down to pair-wised product then sum.
    Softmax is not needed to handle 1 output node, sigmoid can be used. The cost function also comes down to a MSE.

    Understanding 3:
    Another understanding of skip-grams model can be related to vector similarity. Our goal is use two matrices to 
    present the whole vocab. Given a center word and one of the surrounding word, we can present them by two vectors.
    The optimization goal is to maximize the similarity of those two vectors, since they accours close in the training
    sentense. The similarity is calculated as vector dot product.

    Note: [1] provides the detail of skip-gram model calculation, sub-sampling method, and negative-sampling method.

    References: 
    [1] Distributed Representations of Words and Phrases and their Compositionality
        http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    [2] Efficient Estimation of Word Representations in Vector Space
        http://arxiv.org/pdf/1301.3781.pdf
    """
    def __init__(self, input_dim, proj_dim=128,
            init='uniform', activation = 'sigmoid', w_scale=1e-2):
        super(WordContext, self).__init__()
        self.input_dim = input_dim # |v|
        self.proj_dim  = prof_dim # n
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.w_scale = w_scale

        self.input = T.imatrix()
        self.W_w = self.init((input_dim, proj_dim), w_scale)
        self.W_c = self.init((input_dim, proj_dim), w_scale)

        self.params = [self.W_w, self.W_c]


    def get_output(self, train=False):
        X = self.get_input(train)
        w = self.W_w[X[:,0]] # nb_sample x proj_dim, column selection
        c = self.W_c[X[:,1]] # nb_sample x proj_dim, column selection

        dot = T.sum(w * c, axis=1)
        dot = T.reshape(dot, (X.shape[0], 1))
        return self.activation(dot)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "proj_dim": self.proj_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__}










