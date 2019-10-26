import numpy as np
import pandas as pd

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: Tweets
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """

    # Generate vocabulary
    vocab = set()
    text.str.split().apply(vocab.update)

    # Generate lookup tables
    vocab_to_int = {word : ii for ii, word in enumerate(vocab, 1)}
    int_to_vocab = {ii : word for word, ii in vocab_to_int.items()}

    # Add padding special word
    vocab_to_int['<PAD>'] = 0
    int_to_vocab[0] = '<PAD>'

    # return tuple
    return (vocab_to_int, int_to_vocab)
