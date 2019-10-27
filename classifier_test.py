import numpy as np
import torch

from utils import get_loaders
from classifier_utils import HateSpeechClassifier

train_on_gpu = torch.cuda.is_available()

def _test_HateSpeechClassifier():
    batch_size = 20
    sequence_length = 14
    vocab_size = 3
    output_size= 4
    embedding_dim= 25
    hidden_dim = 12
    n_layers = 2
    cnn_params = (5, 3, 1, 1)
    pool_params = (2, 2, 0)
    vocab_to_int = {'banana' : 0, 'apple' : 1, 'orange' : 2}
    embedding_path = "glove/glove.twitter.27B.25d.txt"

    # Initialize model
    test_classifier = HateSpeechClassifier(vocab_size, output_size, embedding_dim,
                                           cnn_params, pool_params, hidden_dim, n_layers,
                                           embedding_path=embedding_path, vocab_to_int=vocab_to_int)

    # create test input
    X_npy = np.random.randint(vocab_size, size=(batch_size, sequence_length))
    X = torch.from_numpy(X_npy)

    # Move to GPU if available
    if(train_on_gpu):
        test_classifier.cuda()
        X = X.cuda()

    # Compute
    hidden = test_classifier.init_hidden(batch_size)
    out, hidden_out = test_classifier(X, hidden)

    # Test output and hidden state shapes
    assert out.shape == (batch_size, output_size)
    assert hidden_out[0].size() == (n_layers, batch_size, hidden_dim)
    assert len(test_classifier.embedding.weight.data.shape) == 2
    assert test_classifier.embedding.weight.data.shape[0] == len(vocab_to_int)

if __name__ == "__main__":

    tweets = np.load("data/tweets.npy")
    labels = np.load("data/hate_original.npy")
    train_on_gpu = torch.cuda.is_available()
    print("Testing on GPU." if train_on_gpu else "Testing on CPU.")

    train_loader, valid_loader, test_loader = get_loaders(tweets, labels, prnt=False)

    print("Testing HateSpeechClassifier class...\n")
    _test_HateSpeechClassifier()

    print("All tests were successful.\n")
