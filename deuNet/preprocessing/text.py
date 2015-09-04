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


