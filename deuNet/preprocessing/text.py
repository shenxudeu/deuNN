"""
Commonly used text preprocessings
"""

import string, sys
import numpy as np

def base_filter():
    """
    return all the punctuations strings we want to remove from the text
    """
    f = string.punctuation
    f = f.replace("'",'') # we want ' in the text
    f += '\t\n'
    return f


def text_to_word_sequence(text, filters=base_filter(), lower=True, split=" "):
    """
    Convert text string to list of words after removing punctuations
    """
    if lower:
        text = text.lower()
    text =text.translate(string.maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [_f for _f in seq if _f]


class Tokenizer(object):
    """
    Main class used to parse raw text files.
    Input texts is a list of documents, each document is a string.
    It will tokenize the raw text and convert it to ints or list of tokens
    """
    def __init__(self, nb_words=None, filters = base_filter(), lower=True, split=" "):
        """
        Inputs:
         - nb_words: int, number of words in vocab, it will take the most common N words
         - filters: string, all punctuations
         - lower: bool, True means convert text to all lower case
         - split: string, how to split tokens
        """
        self.word_counts = {} # words counter
        self.filters = filters
        self.lower = lower
        self.nb_words = nb_words
        self.document_count = 0
        self.split = split

    
    def fit_on_texts(self, texts):
        for text in texts:
            self.document_count += 1
            seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1

            wcounts = list(self.word_counts.items())
            wcounts.sort(key = lambda x: x[1], reverse=True)
            sorted_voc = [wc[0] for wc in wcounts]
            self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
            
    def text_to_sequences(self, texts):
        """
        transform each text in texts to a sequence of integers.
        top 'nb_words' most frequent words will be taken into account.
        Outputs:
            - list, list of sequences.
        """
        self.fit_on_texts(texts)
        res = []
        for vect in self.text_to_sequences_generator(texts):
            res.append(vect)
        return res


    def text_to_sequences_generator(self, texts):
        nb_words = self.nb_words
        for text in texts:
            seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if nb_words and i >= nb_words:
                        pass
                    else:
                        vect.append(i)

            yield vect


