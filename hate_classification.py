import re
from string import punctuation

from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.tokenize import SpaceTokenizer

def hate_classification(hate_tweet):
    '''Receives a hateful tweet.
       Return 3 for directed hate speech and 4 otherwise.'''

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
