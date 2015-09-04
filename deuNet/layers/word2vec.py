"""
word2vec.py: Popular word2vec models

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


