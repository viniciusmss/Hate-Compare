import pandas as pd
import numpy as np
from embedding import get_corpus
from nltk import pos_tag
import gensim
import re
from nltk.corpus import wordnet as wn
from textgenrnn import textgenrnn
#from augment import csv_to_txt
import os

class Augment():

    def __init__(self,
                 method,
                 source_path,
                 target_path,
                 corpus_='none',
                 valid_tags=['NN'],
                 threshold=0.75,
                 x_col='tweet',
                 y_col='class'):
        """
        Constructor Arguments
        method (string):
            valid args:
                'postag': repalces all words of a given POS-tag in the sentence
                    with their most similar word vector from a large pre-trained
                    word embedding
                'threshold': Loads in a pre-trained word embedding and replaces
                    words in a sentence with the word vector of highest cosine
                    similarity
                'generative': trains a two-layer LSTM network to learn the word
                    representations of given class. The network then generates
                    samples of the class by initialising a random start word
                    and following the LSTM's predictions of the next word given
                    the previous sequence
        source_path (string): csv file that is meant to be augmented
        corpus_ (string): Word corpus that the similarity model should take in
            valid args: ['none', 'glove', 'fasttext', 'google']
        x_col (string): column name in csv from samples
        y_col (string): column name in csv for labels
        """

        self.model = get_corpus(corpus_)
        print('Loaded corpus: ', corpus_)
        self.x_col=x_col
        self.y_col=y_col
        self.df=pd.read_csv(source_path)
        self.augmented=pd.DataFrame(columns=[x_col, y_col])
        self.method=method
        self.valid_tags = valid_tags
        self.threshold_ = threshold
        # Go through each row in dataframe
        if method != 'generate':
            for idx, row in self.df.iterrows():
                x = self.preprocess(row[self.x_col])
                y = row[self.y_col]
                if method =='postag':
                    aug_temp = self.postag(x)
                if method =='threshold':
                    aug_temp = self.threshold(x)

            for elem in aug_temp:
                self.augmented.loc[augmented.shape[0]] = [x, y]
        else:
            self.generate('hate', 100)


        self.augmented.to_csv(target_path, encoding='utf-8')


    def preprocess(self, x):
        x = re.sub("[^a-zA-Z ]+", "", x)
        x = x.split()
        return x

    def generate(self):
        print('placeholder')

    def postag(self, x):
        n = 0
        dict = {}
        tags = pos_tag(x)
        for idx, word in enumerate(x):
            if tags[idx][1] in self.valid_tags and word in self.model.wv.vocab:
                replacements = self.model.wv.most_similar(positive=word, topn=3)
                replacements = [elem[0] for elem in replacements]
                dict.update({word:replacements})
                n = len(replacements) if len(replacements) > n else n
        return self.create_augmented_samples(dict, n, x)

    def create_augmented_samples(self, dict, n, x):
        aug_tweets = [x]
        for i in range(n):
            single_augment = x[:]
            for idx, word in enumerate(single_augment):
                if word in dict.keys() and len(dict[word]) >= i+1:
                    single_augment[idx] = dict[word][i]
            single_augment = ' '.join(single_augment)
            aug_tweets.append(single_augment)
        print(aug_tweets)
        return aug_tweets


    def threshold(self, x):
        dict = {}
        n = 0
        tags = pos_tag(x)
        for idx, word in enumerate(x):
            if word in self.model.wv.vocab:
                #get words with highest cosine similarity
                replacements = self.model.wv.most_similar(positive=word, topn=5)
                #keep only words that pass the threshold
                replacements = [replacements[i][0] for i in range(5) if replacements[i][1] > self.threshold_]
                #check for POS tag equality, dismiss if unequal
                replacements = [elem for elem in replacements if pos_tag([elem.lower()])[0][1] == tags[idx][1]]
                dict.update({word:replacements}) if len(replacements) > 0 else dict
                n = len(replacements) if len(replacements) > n else n
        return self.create_augmented_samples(dict, n, x)


    def generate(self, class_, n_, to_file=False):
        """
        Takes in the name of the class and number of samples to be generated.
        Trains a 2-layer LSTM network to generate new samples of given class
        and writes them to the target path
        Args:
            class_(int): integer denoting the class number from dataframe/csv
            n_(int): integer denoting the number of new samples to be created
            toFile(bool): will write to self.target if set to True
        """
        df_class = self.df[self.df[y_col] == class_]
        class_path = 'class_filter.csv'
        df_class.to_csv(class_path) #write temporary csv file to train RNN instance
        textgen = textgenrnn(name=self.method)
        textgen.train_from_file(
            file_path=class_path,
            num_epochs=10,
            batch_size=128,
            new_model=True,
            word_level=True
        )

        #generate new samples
        if to_file:
            textgen.generate_to_file(self.target, n=n_, temperature=1.0)
        else: #will print to screen
            textgen.generate(n_, temperature=1.0)

        #clean up repo
        os.remove(class_path)
        os.remove(self.method+'_config.json')
        os.remove(self.method+'vocab.json')
        os.remove(self.method+'_weights.hdf5')


if __name__ == '__main__':
    Augment('generate', 'preprocessed_data.csv', 'augmented_data.csv', 'google')
