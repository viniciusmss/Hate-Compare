import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


def get_corpus(corpus_):
    """Loads pre-trained word2vec model from src/ directory and
    returns a gensim word2vec object"""
    if corpus_ == 'google':
        return KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                 binary=True)
    if corpus_=='glove':
        return KeyedVectors.load_word2vec_format('glove_gensim_vectors.txt',
                                                 binary=False)
    if corpus_=='fasttext':
        return KeyedVectors.load_word2vec_format('crawl-300d-2M.vec',
                                                 binary=False,
                                                 encoding='UTF-8')




if __name__ == '__main__':
    model = get_corpus('google')
