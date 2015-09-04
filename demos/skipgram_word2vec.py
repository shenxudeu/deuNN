"""
Example of training a skip-gram word2vec distributed representation
"""

import numpy as np
import sys

import pdb

sys.path.append('../../deuNet/')

from deuNet.preprocessing import text
from deuNet.datasets.data_utils import get_file 

path = get_file('sample_text.txt','http://cs.stanford.edu/people/karpathy/char-rnn/pg.txt')

data = open(path,'r').read()
data = [data]
tokenizer = text.Tokenizer()
res = tokenizer.text_to_sequences(data)

pdb.set_trace()
tmp = 1

