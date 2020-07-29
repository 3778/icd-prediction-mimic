#    Copyright 2020, 37.78 Tecnologia Ltda.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
from time import time
import random
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords as STOP_WORDS

import utils

from constants import W2V_DIR, W2V_SIZE, MAX_LENGTH


# Set random seed
random.seed(3778)

class TFIDF: 

    def __init__(self, args):
        self.args = args
        self.tfidf = TfidfVectorizer(stop_words = STOP_WORDS.words('english'), max_features=self.args.max_features)

    def fit(self, dataset): 

        X = dataset.x_train.pipe(utils.preprocessor_tfidf)

        # Fit TF-IDF Transformer 
        self.tfidf.fit(X)

    def transform(self, dataset):

        def transform_subset(X):
            X = X.pipe(utils.preprocessor_tfidf)

            return self.tfidf.transform(X).toarray()

        self.x_train = transform_subset(dataset.x_train)
        self.x_val = transform_subset(dataset.x_val)
        self.x_test = transform_subset(dataset.x_test)

        print('''
            Texts transformed!
        ''')


class W2V:

    def __init__(self, args=None):

        if type(args) == str: # i.e. if name is passed
            self.name = args
            self.init_from_file()
        else:
            self.args = args
            # Instantiate model
            self.model_w2v = Word2Vec(min_count=10, window=5, size=W2V_SIZE, sample=1e-3, negative=5,
                            workers=self.args.workers, sg=self.args.sg, seed=3778)


    def init_from_file(self): # substitutes load_embedding

        # Also load model?
        self.model_w2v = None

        # Load embedding matrix
        with open(f'{W2V_DIR}{self.name}_emb_train_vec{W2V_SIZE}.pkl','rb') as file:
            self.embedding_matrix = pickle.load(file)
        
        # Load row_dict
        with open(f'{W2V_DIR}{self.name}_dict_train_vec{W2V_SIZE}.pkl','rb') as file:
            self.row_dict = pickle.load(file)



    def fit(self, dataset, verbose=1):

        if verbose:
            print('''
            Training embeddings...
            ''')

        token_review = dataset.x_train.pipe(utils.preprocessor)

        # Build vocab over train samples
        self.model_w2v.build_vocab(token_review)

        # We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
        t0 = time()
        for _ in range(5):
            self.model_w2v.train(np.random.permutation(token_review), 
                            total_examples=self.model_w2v.corpus_count, 
                            epochs=self.model_w2v.epochs)
        elapsed=time() - t0
        if verbose: 
            print(f'''
            Time taken for Word2vec training: {elapsed:.2f} seconds.
            ''')


        # Create word embedding matrix
        self.embedding_matrix = self.model_w2v.wv[self.model_w2v.wv.vocab]

        # Create dict for embedding matrix (word <-> row)
        self.row_dict=dict({word:idx for idx,word in enumerate(self.model_w2v.wv.vocab)})

        # Create and map unknown and padding tokens to null
        self.embedding_matrix = np.concatenate((self.embedding_matrix, np.zeros((2,W2V_SIZE))), axis=0)
        self.row_dict['_unknown_'] = len(self.model_w2v.wv.vocab)
        self.row_dict['_padding_'] = len(self.model_w2v.wv.vocab) + 1

        if self.args.reset_stopwords:
            stopwords = STOP_WORDS.words('english')
            for word in self.row_dict:
                if word in stopwords: self.embedding_matrix[self.row_dict[word]] = np.zeros(W2V_SIZE)

        
        if verbose:
            print(f'''
            W2V embedding matrix shape: {self.embedding_matrix.shape}
            ''')


    def transform(self, dataset):

        def transform_X(X):
            return (X
                    .pipe(utils.preprocessor_word2vec)
                    .apply(utils.convert_data_to_index, row_dict=self.row_dict)
                    .apply(lambda x: np.squeeze(pad_sequences([x], padding = 'post', truncating = 'post',
                                                maxlen = MAX_LENGTH, value = self.row_dict['_padding_']))))

        
        self.x_train = np.vstack(transform_X(dataset.x_train).to_list())
        self.x_val = np.vstack(transform_X(dataset.x_val).to_list())
        self.x_test = np.vstack(transform_X(dataset.x_test).to_list())

        print('''
            Texts transformed!
        ''')


    def save_embedding(self, dataset_name='MIMIC'):
        
        # Save embedding layer and row dict
        with open(f'{W2V_DIR}{dataset_name}_emb_train_vec{W2V_SIZE}.pkl', 'wb') as file:
            pickle.dump(self.embedding_matrix, file)

        with open(f'{W2V_DIR}{dataset_name}_dict_train_vec{W2V_SIZE}.pkl', 'wb') as file:
            pickle.dump(self.row_dict, file)

        
        # Save Word2Vec model
        self.model_w2v.save(f'{W2V_DIR}w2v_model.model')


    def load_embedding(self, dataset_name='MIMIC'):

        # Load embedding matrix
        with open(f'{W2V_DIR}{dataset_name}_emb_train_vec{W2V_SIZE}.pkl','rb') as file:
            self.embedding_matrix = pickle.load(file)
        
        # Load row_dict
        with open(f'{W2V_DIR}{dataset_name}_dict_train_vec{W2V_SIZE}.pkl','rb') as file:
            self.row_dict = pickle.load(file)

