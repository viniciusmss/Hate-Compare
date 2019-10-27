import re
import html
import numpy as np
import pandas as pd
import torch

from string import punctuation
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.tokenize import SpaceTokenizer
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader

def preprocess(text_string):
    '''

    Heavily drawn from Davidson et al. (2017).
    '''

    # Casing should not make a difference in our case
    text_string = text_string.lower()

    # Regex
    html_pattern = r'(&(?:\#(?:(?:[0-9]+)|[Xx](?:[0-9A-Fa-f]+))|(?:[A-Za-z0-9]+));)'
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'

    # First, add space surrounding HTML entities
    text_string = re.sub(html_pattern, r' \1 ', text_string)

    # Now, if we wish to find hashtags, we have to unescape HTML entities
    text_string = html.unescape(text_string)

    # From Udacity TV script generation project
    # Replace some punctuation by dedicated tokens
    symbol_to_token = {
        '.' : '||Period||',
        ',' : '||Comma||',
        '"' : '||Quotation_Mark||',
        ';' : '||Semicolon||',
        '!' : '||Exclamation_Mark||',
        '?' : '||Question_Mark||',
        '(' : '||Left_Parenthesis||',
        ')' : '||Right_Parenthesis||',
        '-' : '||Dash||',
        '\n' : '||Return||'
    }

    # Next, find URLs
    text_string = re.sub(giant_url_regex, ' URLHERE ', text_string)

    # Then, tokenize punctuation
    for key, token in symbol_to_token.items():
        text_string = text_string.replace(key, ' {} '.format(token))

    # Finally, remove spaces and find mentions and hashtags
    text_string = re.sub(hashtag_regex, ' HASHTAGHERE ', text_string)
    text_string = re.sub(mention_regex, ' MENTIONHERE ', text_string)
    text_string = re.sub(space_pattern, ' ', text_string)

    return text_string

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: Tweets
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)

    Drawn from Udacity. (2019).
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

def create_pad_fn(max_length):

    def pad_tweets(tweet, max_length=max_length):
        # Do not cut tweet short if it's too long

        # Retrieve tweet word count
        word_count = len(tweet.split())

        # Check how much padding will be needed
        n = max_length - word_count if word_count < max_length else 0

        # Pad tweet
        padded_tweet = ''.join(['<PAD> '] * n + [tweet])

        return padded_tweet

    return pad_tweets

def pad_tweets(tweet, max_length=10):
    # Do not cut tweet short if it's too long

    # Retrieve tweet word count
    word_count = len(tweet.split())

    # Check how much padding will be needed
    n = max_length - word_count if word_count < max_length else 0

    # Pad tweet
    padded_tweet = ''.join(['<PAD> '] * n + [tweet])

    return padded_tweet

def hate_classification(hate_tweet):
    '''
    Receives a hateful tweet.
    Return 3 for directed hate speech and 4 otherwise.

    I do not implement co-reference resolution since a single NE is sufficient for directed hate speech.

    Partly drawn from imanzabet. (2017).
    '''

    if bool(hate_tweet.count("MENTIONHERE")): return(3)

    # Remove tokens since they will oncused the POS tagger
    token_regex = '\|\|\w+\|\|'
    hate_tweet = re.sub(token_regex, "", hate_tweet)

    # URLHERE is considered a proper noun by the pos tagger.
    # Remove them before checking for proper nouns
    no_punct_hate = ''.join([char for char in hate_tweet if char not in punctuation])
    no_URL_hate = ' '.join([token for token in no_punct_hate.split() if token != "URLHERE"])
    has_NE = False
    for sent in sent_tokenize(no_URL_hate):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                return(3)  # Named Entity found

    return(4)

def change_hate_labels(tweets, raw_labels):
    ''' Change hate speech labels (0) to directed (3) / generalized labels (4)
        Shifts class numbers to the left so that class labels start from zero.
        Returned labels:

            (0) : Offensive
            (1) : Neither
            (2) : Directed hate speech
            (3) : Generalized hate speech

    '''
    labels = raw_labels.copy()

    for i, (tweet, label) in enumerate(zip(tweets, raw_labels)):

        if label == 0:  # If hate speech
            labels[i] = hate_classification(tweet)

    return labels - 1

def get_loaders(features, labels, train_frac=0.8, remain_split_frac=0.5,
                batch_size=64, prnt=True):

    features, labels = shuffle(features, labels)

    ## split data into training, validation, and test data (features and labels, x and y)
    split_idx = int(features.shape[0]*train_frac)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = labels[:split_idx], labels[split_idx:]

    test_idx = int(len(remaining_x)*remain_split_frac)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    if prnt:
        ## print out the shapes of your resultant feature data
        print("\t\t\tFeature Shapes:")
        print("Train set: \t\t{}".format(train_x.shape),
              "\nValidation set: \t{}".format(val_x.shape),
              "\nTest set: \t\t{}".format(test_x.shape))

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return(train_loader, valid_loader, test_loader)
