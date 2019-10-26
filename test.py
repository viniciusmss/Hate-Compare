import numpy as np
import pandas as pd

from preprocess import preprocess
from create_lookup_tables import create_lookup_tables


def _test_preprocess():

    assert " HASHTAGHERE " == preprocess("#iam1hashtag")
    assert " URLHERE " == preprocess("https://seminar.minerva.kgi.edu")
    assert " MENTIONHERE " == preprocess("@vinimiranda")
    assert ' ' == preprocess("        ")
    assert " & MENTIONHERE URLHERE HASHTAGHERE " == \
        preprocess("&amp;@vinimiranda    https://seminar.minerva.kgi.edu     #minerva    ")

def _test_lookup_tables():

    text = pd.Series(["this is a toy", "I mean not really a toy", "I mean a toy vocabulary"])
    vocab_to_int, int_to_vocab = create_lookup_tables(text)

    # Make sure the dicts make the same lookup
    missmatches = [(word, id, id, int_to_vocab[id]) for word, id in vocab_to_int.items() if int_to_vocab[id] != word]

    assert not missmatches,\
        'Found {} missmatche(s). First missmatch: vocab_to_int[{}] = {} and int_to_vocab[{}] = {}'.format(len(missmatches),
                                                                                                          *missmatches[0])

if __name__ == "__main__":
    print("Testing preprocessing function...")
    _test_preprocess()

    print("Testing create lookup table function...")
    _test_lookup_tables()

    print("All tests were successful.\n")
