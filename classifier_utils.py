import gensim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

train_on_gpu = torch.cuda.is_available()

class HateSpeechClassifier(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, cnn_params, pool_params,
                 hidden_dim, n_layers, dropout=0.5, embedding_path=None, vocab_to_int=None):
        """
        TO BE RESTATED
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param cnn_params: A 4-element tuple containing the number
            of feature maps, kernel size, stride and padding of a Conv1D layer.
        :param pool_params: A 3-element tuple containing the kernel size, stride and padding of a MaxPool1D layer.
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(HateSpeechClassifier, self).__init__()

        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_to_int = vocab_to_int

        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_path is not None:
            self.set_pretrained_weights(embedding_path)

        self.conv = nn.Conv1d(embedding_dim, *cnn_params)

        self.pool = nn.MaxPool1d(*pool_params)

        n_maps, _, _, _ = cnn_params
        self.lstm = nn.LSTM(n_maps, hidden_dim, n_layers,
                            dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(hidden_dim, output_size)


    def forward(self, nn_input, hidden, test_print=False):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function
        batch_size = nn_input.size(0)

        # embeddings
        nn_input = nn_input.long()
        embeds = self.embedding(nn_input)

        # Change axes. embedding_dim (in_channels) should be in the middle
        # [batch_size, seq_length, embedding_dim] -> [batch_size, embedding_dim, seq_length]
        embeds_t = embeds.permute(0, 2, 1)

        # conv
        conv_out = self.conv(embeds_t)

        # pool
        pool_out = self.pool(F.relu(conv_out))

        # Change axes. lstm expects features to be the last channel
        # [batch_size, n_maps, down_sampled_seq] -> [batch_size, down_sampled_seq, n_maps]
        pool_out_t = pool_out.permute(0, 2, 1)

        # lstm
        lstm_out, hidden = self.lstm(pool_out_t, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        # out = self.dropout(lstm_out)
        fc_out = self.fc(lstm_out)

        # reshape to be batch_size first
        fc_out_t = fc_out.view(batch_size, -1, self.output_size)

        out = fc_out_t[:, -1] # get last batch of labels

        if test_print:
            print("nn_input.\nexpected : [batch_size, seq_length].\nshape: {}\n".format(nn_input.shape))
            print("embeds.\nexpected : [batch_size, seq_length, embedding_dim].\nshape: {}\n".format(embeds.shape))
            print("embeds_t.\nexpected : [batch_size, embedding_dim, seq_length].\nshape: {}\n".format(embeds_t.shape))
            print("conv_out.\nexpected : [batch_size, n_maps, seq_length].\nshape: {}\n".format(conv_out.shape))
            print("pool_out.\nexpected : [batch_size, n_maps, down_sampled_seq].\nshape: {}\n".format(pool_out.shape))
            print("pool_out_t.\nexpected : [batch_size, down_sampled_seq, n_maps].\nshape: {}\n".format(pool_out_t.shape))
            print("lstm_out.\nexpected : [batch_size, down_sampled_seq, hidden_dim].\nshape: {}\n".format(lstm_out.shape))
            print("lstm_out.\nexpected : [batch_size * down_sampled_seq, hidden_dim].\nshape: {}\n".format(lstm_out.shape))
            print("fc_out.\nexpected : [batch_size * down_sampled_seq, output_dim].\nshape: {}\n".format(fc_out.shape))
            print("fc_out_t.\nexpected : [batch_size, down_sampled_seq, output_dim].\nshape: {}\n".format(fc_out_t.shape))
            print("out.\nexpected : [batch_size, output_dim].\nshape: {}\n".format(out.shape))

        # return one batch of output word scores and the hidden state
        return out, hidden

    def set_pretrained_weights(self, embedding_path, pnt=True):

        if not hasattr(self, 'word2vec_model'):
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path)

        # Check whether the pretrained model has the correct dimensionality
        assert len(self.word2vec_model["human"]) == self.embedding_dim

        # Create the lookup table
        embedding_weights = np.zeros((len(self.vocab_to_int), self.embedding_dim))

        n = 0  # For each word in the dictionary
        for word, value in self.vocab_to_int.items():

            try:
                # Find its embeddings
                embedding_weights[value] = self.word2vec_model[word]

            except:
                # Or report that it's missing
                n += 1

        if pnt: print("{} words in the vocabulary have no pre-trained embedding.".format(n))

        device = "cuda:0" if train_on_gpu else "cpu"
        embedding_weights = torch.Tensor(embedding_weights).type(torch.FloatTensor).to(device)
        self.embedding.weight = nn.Parameter(embedding_weights)

    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function

        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

def forward_back_prop(model, optimizer, criterion, inp, target, hidden, clip=5):
    """
    Forward and backward propagation on the neural network
    :param model: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """

    batch_size = inp.size(0)
    target = target.type(torch.LongTensor)

    # move data to GPU, if available
    if train_on_gpu:
        inp, target = inp.cuda(), target.cuda()

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    hidden = tuple([each.data for each in hidden])

    # zero accumulated gradients
    model.zero_grad()

    # get the output from the model
    output, hidden = model(inp, hidden)

    # perform backpropagation and optimization
    # calculate the loss and perform backprop
    loss = criterion(output, target)

    try:
        loss.backward()

    except RuntimeError:
        fn = lambda x, y : print('{} : {}'.format(x, y.shape))
        fn('output', output)
        fn('target', target)
        fn('loss', loss.item())

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden
