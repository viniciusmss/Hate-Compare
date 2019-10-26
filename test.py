import numpy as np
import pandas as pd

from preprocess import preprocess
from create_lookup_tables import create_lookup_tables
from padding import create_pad_fn, pad_tweets


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

def _test_pad_tweets():

    assert pad_tweets('hi', 0) == 'hi'
    assert pad_tweets('hi', 1) == 'hi'
    assert pad_tweets('hi', 2) == '<PAD> hi'
    assert len(pad_tweets('hi', 10).split()) == 10
    assert len(pad_tweets('hi', 100).split()) == 100
    assert pad_tweets('this sentence is a bit longer', 1) == 'this sentence is a bit longer'

if __name__ == "__main__":

    df = pd.read_csv("./data/labeled_data.csv", index_col=0)
    raw_tweets = df.tweet
    raw_labels = df["class"].values

    print("Testing preprocessing function...\n")
    _test_preprocess()

    tweets = raw_tweets.map(preprocess)
    df["clean_tweet"] =  tweets # Get cleaned tweets
    df["word_count"] = df.clean_tweet.apply(lambda x : len(x.split()))  # Get their word count
    # Remove outliers
    old_tweet = df.loc[df.word_count == df.word_count.max(),].tweet.values[0]
    new_tweet = old_tweet[:old_tweet.find("\r")]
    df.loc[df.word_count == df.word_count.max(), "tweet"] = new_tweet
    df.loc[df.word_count == df.word_count.max(), "clean_tweet"] = preprocess(new_tweet)
    df.loc[df.word_count == df.word_count.max(), "word_count"] = len(preprocess(new_tweet).split())

    print("Testing create lookup table function...\n")
    _test_lookup_tables()

    vocab_to_int, int_to_vocab = create_lookup_tables(tweets)

    print("Testing padding function...\n")
    _test_pad_tweets()

    print("All tests were successful.\n")
