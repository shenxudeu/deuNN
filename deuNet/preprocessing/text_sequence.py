"""
Text Sequency processing
includes word2vec Negative Samplings
"""

import numpy as np
import random

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
    Pad sequences to the same length, which is the length of longest one.
    Sequences are presented by ints.
    
    Inputs:
        - sequences: list, list of list of ints
        - maxlen: int, max length of a sequence
        - dtype: string, dtype of the sequence
        - padding: string, "pre" or "post", pre or post padding
        - truncating: string, "pre" or "pos", pre or post truncating
        - value: int, padding value
    Outpus:
        - padded sequences, list of list of ints
    """

    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)

    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncting == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not supported"% truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not supported"% truncating)

    return x


def make_subsampling_table(size, sampling_factor = 1e-5):
    """
    The subsampling table is used to select training words.
    Because in very large corpora, the most frequent words can
    easily occur hundreds of millions of times, such words 
    usually provide less information than the rare words.

    Reference:
    http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    Sector 2.3 - subsampling of frequent words
    """
    gamma = 0.577
    rank = np.array(list(range(size)))
    rank[0] = 1
    inv_fq = rank * (np.log(rank) + gamma) + 0.5 - 1./(12.*rank)
    f = sampling_factor * inv_fq
    return np.minimum(1., f/np.sqrt(f))


def skipgrams(sequence, vocabulary_size,
        window_size=4, negative_samples=1., shuffle=True,
        subsampling_table=None)
    """
    Generate training couples for skipgrams wrod2vec embedding
    The generated dataset has two lists:
        - couples: a list of list, sublist has length of 2, first is the center word, second is from surrounding word
        - labels: a list, 1 means this is a positive sample, 0 means this is a negative sample
    
    Inputs:
        - sequence: list of int, list of indices of words
        - vocabulary_size: int, |v|
        - window_size: int, half of the surrounding words length. The surrounding windows is [i-window_size, i+window_size]
        - negative_samples: float, negative samples length (in percentage), 1. means the same size of positive samples
        - shuffle: bool, True means shuffle the traning sample order
        - subsampling_table: np.array, the array indicating subsampling prob density, generated from make_subsampling_table()
    """
    couples = []
    labels = []

    # loop each word in the sequence and construct positive training data
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if subsampling_table is not None:
            if subsampling_table[wi] < random.random():
                continue

        window_start = max(0, 1-window_size)
        window_end   = min(len(sequence), i + window_size +1)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                labels.append(1)
    
    if negative_samples > 0:
        nb_negative_samples = int(len(labels)*negative_samples)
        words = [c[0] for c in couples] # all center words
        random.shuffle(words)

        couples += [[words[i%len(words)], random.randint(1, vocabulary_size-1)] for i in range(nb_negative_samples)]
        labels  += [0] * nb_negative_samples
    
    if shuffle:
        seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels

