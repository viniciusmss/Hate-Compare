import gensim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Note: Helper function here are heavily inspired by the code base of 
# Udacity's Deep Learning Nanodegree. Check it out!
# Udacity. (2019). TV Script Generation. In Deep Learning (PyTorch). Retrieved from https://github.com/udacity/deep-learning-v2-pytorch

class HateSpeechClassifier(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, cnn_params, pool_params,
                 hidden_dim, n_layers, p_lstm_dropout= 0, p_dropout=0, embedding_path=None,
                 vocab_to_int=None, train_on_gpu=False):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param cnn_params: A 4-element tuple containing the number
            of feature maps, kernel size, stride and padding of a Conv1D layer.
        :param pool_params: A 3-element tuple containing the kernel size, stride and padding of a MaxPool1D layer.
        :param hidden_dim: The size of the hidden layer outputs
        :param n_layers: The number of stacked LSTM layers
        :param p_lstm_dropout: The dropout between stacked LSTM layers [default: 0] 
        :param p_dropout: Probability of dropout before the classification layer [default: 0]
        :param embedding_path: Path to pretrained embedding model (e.g., GloVe) [default: None]
        :param vocab_to_int: If using pretrained model, a word-to-integer dictionary is necessary
            to construct the embedding [default: 0]
        :param train_on_gpu: Whether to train the network on GPU [default: False]
        """
        super(HateSpeechClassifier, self).__init__()

        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_to_int = vocab_to_int
        self.train_on_gpu = train_on_gpu

        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_path is not None:
            self.set_pretrained_weights(embedding_path)

        self.conv = nn.Conv1d(embedding_dim, *cnn_params)

        self.pool = nn.MaxPool1d(*pool_params)

        n_maps, _, _, _ = cnn_params
        self.lstm = nn.LSTM(n_maps, hidden_dim, n_layers,
                            dropout=p_lstm_dropout, batch_first=True)

        self.dropout = nn.Dropout(p_dropout)

        self.fc = nn.Linear(hidden_dim, output_size)


    def forward(self, nn_input, hidden, test_print=False):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :param test_print: When debugging, set to true to show expected and actual shapes of tensors.
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        
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
        lstm_out = self.dropout(lstm_out)
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

        print("Setting pretrained embedding weights...")

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

        device = "cuda:0" if self.train_on_gpu else "cpu"
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

        if (self.train_on_gpu):
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
    if model.train_on_gpu:
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

def train_classifier(model, batch_size, optimizer, criterion, n_epochs, train_loader, valid_loader,
                     show_every_n_batches=10, try_load = False, save_path = None):

    # Load model previously trained if availabale
    if try_load:
        try:
            model.load_state_dict(torch.load(save_path))
            return model
        except:
            pass

    # n steps
    steps = 0

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        # initialize variables to monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################

        # initialize hidden state
        hidden = model.init_hidden(batch_size)

        # Set model for training
        model.train()

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break

            # forward, back prop
            loss, hidden = forward_back_prop(model, optimizer, criterion, inputs, labels, hidden)

            # record loss
            train_loss += loss

            if batch_i % show_every_n_batches == 0:
                print("Epoch: {}/{}. \tBatch: {}/{}.\t Avg. Training Loss: {}".format(epoch_i,
                                                                                      n_epochs,
                                                                                      batch_i,
                                                                                      len(train_loader),
                                                                                      train_loss/batch_i))

        ######################
        # validate the model #
        ######################

        valid_loss = 0.0
        correct = 0.0
        total = 0.0

        # Initialize hidden state
        valid_hidden = model.init_hidden(batch_size)

        # Set model for evaluation
        model.eval()

        for batch_i, (inputs, labels) in enumerate(valid_loader, 1):

            # make sure you iterate over completely full batches, only
            n_batches = len(valid_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break

            labels = labels.type(torch.LongTensor)

            # move data to GPU, if available
            if model.train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state
            valid_hidden = tuple([each.data for each in valid_hidden])

            # get the output from the model
            output, valid_hidden = model(inputs, valid_hidden)

            # calculate the loss
            loss = criterion(output, labels)

            # update running validation loss
            valid_loss += loss

            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]

            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(labels.data.view_as(pred))).cpu().numpy())
            total += inputs.size(0)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader)
        valid_loss = valid_loss/len(valid_loader)
        acc = 100. * correct / total

        # print validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \t Accuracy: {:.6f}\n'.format(
            epoch_i,
            train_loss,
            valid_loss,
            acc
            ))

        # save model if validation loss has decreased
        if save_path is not None:
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(
                valid_loss_min,
                valid_loss))
                torch.save(model.state_dict(), save_path)
                valid_loss_min = valid_loss

    # returns a trained classifier
    return model

def batch_test(model, batch_size, test_loader, criterion, prnt=True):

    # Get test data loss and accuracy
    test_loss = 0 # track loss
    num_correct = 0
    total = 0
    y_pred, y_true = [], []

    # init hidden state
    test_hidden = model.init_hidden(batch_size)

    model.eval()
    # iterate over test data
    for batch_i, (inputs, labels) in enumerate(test_loader, 1):

        # make sure you iterate over completely full batches, only
        n_batches = len(test_loader.dataset)//batch_size
        if(batch_i > n_batches):
            break

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        test_hidden = tuple([each.data for each in test_hidden])

        if(model.train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # get predicted outputs
        output, test_hidden = model(inputs, test_hidden)

        # Accumulate loss
        test_loss += criterion(output, labels)

        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]

        # compare predictions to true label
        num_correct += np.sum(np.squeeze(pred.eq(labels.data.view_as(pred))).cpu().numpy())
        total += inputs.size(0)

        # Save prediction and labels
        y_pred += list(pred.squeeze().cpu().numpy())
        y_true += list(labels.data.cpu().numpy())

    test_acc = num_correct/len(test_loader.dataset)

    if prnt:
        # -- stats! -- ##
        # avg test loss
        print("Test loss: {:.3f}".format(test_loss/len(test_loader)))

        # accuracy over all test data
        print("Test accuracy: {:.1f}%".format(100*test_acc))

    return(test_loss, test_acc, y_true, y_pred)

def plot_confusion_matrices(y_true, y_pred, cmap=plt.cm.Blues):

    if max(y_true) == 2:
        class_names = np.array(["Hate Speech", "Offensive","Neither"])
    else:
        class_names = np.array(["Offensive","Neither","Dir. Hate","Gen. Hate"])

    def plot_confusion_matrix(normalize, title, class_names,
                              cmap=cmap):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:

            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        class_names = class_names[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=class_names, yticklabels=class_names,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax


    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(normalize=False, class_names=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(normalize=True, class_names=class_names,
                          title='Normalized confusion matrix')

    plt.show()
