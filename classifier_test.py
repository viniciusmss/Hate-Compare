import json
import torch
import numpy as np
import pandas as pd
from torch import nn
from unittest.mock import MagicMock, patch

from utils import get_loaders
from classifier_utils import HateSpeechClassifier, forward_back_prop, train_classifier

# Note: Helper function here are heavily inspired by the code base of 
# Udacity's Deep Learning Nanodegree. Check it out!
# Udacity. (2019). TV Script Generation. In Deep Learning (PyTorch). Retrieved from https://github.com/udacity/deep-learning-v2-pytorch

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

class _TestNN(torch.nn.Module):
    def __init__(self, input_size, output_size, train_on_gpu):
        super(_TestNN, self).__init__()
        self.decoder = torch.nn.Linear(input_size, output_size)
        self.forward_called = False
        self.train_on_gpu = train_on_gpu

    def forward(self, nn_input, hidden):
        self.forward_called = True
        output = self.decoder(nn_input)

        return output, hidden

def _test_forward_back_prop(classifierNN, forward_back_prop, train_on_gpu):
    batch_size = 20
    sequence_length = 14
    input_size = 20
    output_size= 4
    embedding_dim= 16
    hidden_dim = 12
    n_layers = 2
    cnn_params = (5, 3, 1, 1)
    pool_params = (2, 2, 0)
    learning_rate = 0.01

    model = classifierNN(input_size, output_size, embedding_dim,
                         cnn_params, pool_params, hidden_dim, n_layers,
                         train_on_gpu=train_on_gpu)

    mock_decoder = MagicMock(wraps=_TestNN(input_size, output_size, train_on_gpu))
    if train_on_gpu:
        mock_decoder.cuda()

    mock_decoder_optimizer = MagicMock(wraps=torch.optim.Adam(mock_decoder.parameters(), lr=learning_rate))
    mock_criterion = MagicMock(wraps=torch.nn.CrossEntropyLoss())

    with patch.object(torch.autograd, 'backward', wraps=torch.autograd.backward) as mock_autograd_backward:
        inp = torch.FloatTensor(np.random.rand(batch_size, input_size))
        target = torch.LongTensor(np.random.randint(output_size, size=batch_size))

        hidden = model.init_hidden(batch_size)

        loss, hidden_out = forward_back_prop(mock_decoder, mock_decoder_optimizer, mock_criterion, inp, target, hidden)

    assert (hidden_out[0][0]==hidden[0][0]).sum()==batch_size*hidden_dim
    assert mock_decoder.zero_grad.called or mock_decoder_optimizer.zero_grad.called, 'Didn\'t set the gradients to 0.'
    assert mock_decoder.forward_called, 'Forward propagation not called.'
    assert mock_autograd_backward.called, 'Backward propagation not called'
    assert mock_decoder_optimizer.step.called, 'Optimization step not performed'
    assert type(loss) == float, 'Wrong return type. Expected {}, got {}'.format(float, type(loss))

def _test_train_classifier(*args):

    model = train_classifier(*args, try_load = False, save_path = None)

    assert isinstance(model, HateSpeechClassifier)

if __name__ == "__main__":

    tweets = np.load("data/tweets.npy")
    labels = np.load("data/hate_original.npy")
    with open('vocab_to_int.json', 'r') as fp:
        vocab_to_int = json.load(fp)
    with open('int_to_vocab.json', 'r') as fp:
        int_to_vocab = json.load(fp)

    train_on_gpu = torch.cuda.is_available()
    print("Testing on GPU." if train_on_gpu else "Testing on CPU.")

    train_loader, valid_loader, test_loader = get_loaders(tweets, labels, prnt=False)

    print("Testing HateSpeechClassifier class...\n")
    _test_HateSpeechClassifier()

    print("Testing forward back propagation function...\n")
    _test_forward_back_prop(HateSpeechClassifier, forward_back_prop, train_on_gpu)

    sequence_length = tweets.shape[1]  # number of words in a sequence
    n_epochs = 1
    learning_rate = 0.01
    vocab_size = len(vocab_to_int)
    output_size = pd.Series(labels).nunique()
    embedding_dim = 10
    hidden_dim = 16
    batch_size = 64
    n_layers = 1
    show_every_n_batches = tweets.shape[0] + 1
    cnn_params = (32, 25, 1, 4)
    pool_params = (4, 4, 0)

    print("Instantiating model...\n")
    model = HateSpeechClassifier(vocab_size, output_size, embedding_dim,
                                 cnn_params, pool_params, hidden_dim, n_layers,
                                 dropout=0.5, vocab_to_int=vocab_to_int,
                                 train_on_gpu=train_on_gpu)

    if model.train_on_gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Testing training function...\n")
    _test_train_classifier(model, batch_size, optimizer, criterion, n_epochs,
                           train_loader, valid_loader, show_every_n_batches)


    print("All tests were successful.\n")
