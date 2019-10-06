from data_handler import get_data
import sys
import numpy as np
import pdb, json
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
import pdb
import pickle
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import codecs
import operator
import gensim, sklearn
from collections import defaultdict
from batch_gen import batch_gen
from my_tokenizer import glove_tokenize
import xgboost as xgb

### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
label_map = {
        'none': 0,
        'racism': 1,
        'sexism': 2
    }
tweet_data = get_data()
for tweet in tweet_data:
    texts.append(tweet['text'].lower())
    labels.append(label_map[tweet['label']])
print('Found %s texts. (samples)' % len(texts))


# logistic, gradient_boosting, random_forest, svm, tfidf_svm_linear, tfidf_svm_rbf
word_embed_size = 200
GLOVE_MODEL_FILE = str(sys.argv[1])
EMBEDDING_DIM = int(sys.argv[2])
MODEL_TYPE=sys.argv[3]
MODEL_FILE=sys.argv[4]
print 'Embedding Dimension: %d' %(EMBEDDING_DIM)
print 'GloVe Embedding: %s' %(GLOVE_MODEL_FILE)

word2vec_model1 = np.load(MODEL_FILE)
word2vec_model1 = word2vec_model1.reshape((word2vec_model1.shape[1], word2vec_model1.shape[2]))
f_vocab = open('vocab_fast_text', 'r')
vocab = json.load(f_vocab)
word2vec_model = {}
for k,v in vocab.iteritems():
    word2vec_model[k] = word2vec_model1[int(v)]
del word2vec_model1


SEED=42
MAX_NB_WORDS = None
VALIDATION_SPLIT = 0.2


# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}


def select_tweets_whose_embedding_exists():
    # selects the tweets as in mean_glove_embedding method
    # Processing

    tweet_return_file = "cnn_tweets.pickle"

    # Load if pickled files are available
    try:
        tweet_return = pickle.load(open(tweet_return_file, "rb"))
        print "Tweets loaded from pickled file."

    # Create and save otherwise
    except (OSError, IOError) as e:

        print "Loading tweets with embeddings available..."
        tweets = get_data()
        tweet_return = []
        for tweet in tweets:
            _emb = 0
            words = TOKENIZER(tweet['text'].lower())
            for w in words:
                if w in word2vec_model:  # Check if embeeding there in GLove model
                    _emb+=1
            if _emb:   # Not a blank tweet
                tweet_return.append(tweet)

        pickle.dump(tweet_return, open(tweet_return_file, "wb"))
    print 'Tweets selected:', len(tweet_return)
    return tweet_return


def gen_data():
    # In this function, for all accepted tweets, we turn them into an
    # embedding of EMBEDDING_DIM. We then sum the embeddings of all
    # words within the tweet that have an embedding and divide
    # by the number of words. Hence, the final embedding of the tweet
    # will be the average of the embeddings of its words.

    X_file = "nn_X.pickle"
    y_file = "nn_y.pickle"

    # Load if pickled files are available
    try:
        X = pickle.load(open(X_file, "rb"))
        y = pickle.load(open(y_file, "rb"))
        print "Features and labels loaded from pickled files."

    # Create and save otherwise
    except (OSError, IOError) as e:
        print "Creating features and labels..."

        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 2
                }

        X, y = [], []
        for tweet in tweets:
            words = glove_tokenize(tweet['text']) # .lower()
            emb = np.zeros(word_embed_size)
            for word in words:
                try:
                    emb += word2vec_model[word]
                except:
                    pass
            emb /= len(words)
            X.append(emb)
            y.append(y_map[tweet['label']])

        X = np.array(X)
        y = np.array(y)

        pickle.dump(X, open(X_file, "wb"))
        pickle.dump(y, open(y_file, "wb"))

    return X, y


def get_model(m_type=None):
    if not m_type:
        print 'ERROR: Please provide a valid method name'
        return None

    if m_type == 'logistic':
        logreg = LogisticRegression()
    elif m_type == "gradient_boosting":
        #logreg = GradientBoostingClassifier(n_estimators=10)
        logreg = xgb.XGBClassifier(nthread=-1)
    elif m_type == "random_forest":
        logreg = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    elif m_type == "svm_rbf":
        logreg = SVC(class_weight="balanced", kernel='rbf')
    elif m_type == "svm_linear":
        logreg = LinearSVC(class_weight="balanced")
    else:
        print "ERROR: Please specify a correst model"
        return None

    return logreg


def classification_model(X, Y, model_type="logistic"):
    NO_OF_FOLDS=10
    X, Y = shuffle(X, Y, random_state=SEED)
    print "Model Type:", model_type

    scorers = ['precision_weighted', 'recall_weighted', 'f1_weighted']
    scores = cross_validate(get_model(model_type), X, Y, cv=NO_OF_FOLDS, scoring=scorers, verbose=1, n_jobs=-2)

    scores1 = scores["test_precision_weighted"]
    scores2 = scores["test_recall_weighted"]
    scores3 = scores["test_f1_weighted"]

    print "Precision(avg): %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std() * 2)
    print "Recall(avg): %0.3f (+/- %0.3f)" % (scores2.mean(), scores2.std() * 2)
    print "F1-score(avg): %0.3f (+/- %0.3f)" % (scores3.mean(), scores3.std() * 2)

if __name__ == "__main__":

    #filter_vocab(20000)

    tweets = select_tweets_whose_embedding_exists()
    X, Y = gen_data()

    classification_model(X, Y, MODEL_TYPE)
    pdb.set_trace()
