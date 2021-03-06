import numpy as np
import pandas as pd
import torch

from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.tokenize import SpaceTokenizer

from utils import preprocess, create_lookup_tables, create_pad_fn, pad_tweets,\
                  hate_classification, change_hate_labels, get_loaders

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

def _test_hate_classification():
    assert hate_classification("MENTIONHERE") == 3
    assert hate_classification("Karen is absolutely crazy") == 3
    assert hate_classification("Karen is his sister. She's absolutely crazy") == 3
    assert hate_classification("They should all be sent to Mexico") == 3
    assert hate_classification("They should all leave the country") == 4
    assert hate_classification("some hate speech stuff") == 4
    assert hate_classification("") == 4

def _test_hate_labels(tweets, raw_labels):
    labels = change_hate_labels(tweets, raw_labels)

    assert 4 not in pd.Series(labels).value_counts().index
    assert 2 in pd.Series(labels).value_counts().index
    assert 3 in pd.Series(labels).value_counts().index

def _test_get_loaders(features, labels):

    train_loader, valid_loader, test_loader = get_loaders(features, labels, prnt=False)
    train_x, train_y = iter(train_loader).next()
    valid_x, valid_y = iter(valid_loader).next()
    test_x, test_y = iter(test_loader).next()

    assert isinstance(train_loader, torch.utils.data.dataloader.DataLoader)
    assert isinstance(valid_loader, torch.utils.data.dataloader.DataLoader)
    assert isinstance(test_loader, torch.utils.data.dataloader.DataLoader)
    assert train_x.shape == valid_x.shape == test_x.shape
    assert train_y.shape == valid_y.shape == test_y.shape

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

    MAX_LENGTH = df.word_count.max()
    pad_tweets = create_pad_fn(MAX_LENGTH)
    df["padded_tweets"] = df.clean_tweet.map(pad_tweets)

    print("Testing hate classification function...\n")
    _test_hate_classification()

    print("Testing change hate labels function...\n")
    _test_hate_labels(tweets, raw_labels)

    tweets_ints = np.array([[vocab_to_int[word] for word in tweet.split()] for tweet in df.padded_tweets.values])

    print("Testing get loaders function...\n")
    _test_get_loaders(tweets_ints, raw_labels)

    print("All tests were successful.\n")
