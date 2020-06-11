
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

from constants import W2VDIR, W2V_SIZE, MAX_LENGTH



# FIX random seed

class TFIDF: # not sure it will handle the pandas objects

    def __init__(self, args):
        self.args = args

    def fit(self, X): 

        X = X.pipe(utils.preprocessor_tfidf)

        # Instantiate TF-IDF Transformer (maybe to this in init?)
        self.tfidf = TfidfVectorizer(stop_words = STOP_WORDS.words('english'), max_features=self.args.max_features)

        self.tfidf.fit(X)


    def transform(self, X):

        return self.tfidf.transform(X.pipe(utils.preprocessor_tfidf))


    def save(self):
        pass

    def load(self):
        pass


class W2V:

    def __init__(self, args):
        self.args = args

    def fit(self, X, verbose=1): #ave=True or def save()?

        token_review = X.pipe(utils.preprocessor)

        # Instantiate model (maybe do this in init?)
        self.model_w2v = Word2Vec(min_count=10, window=5, size=W2V_SIZE, sample=1e-3, negative=5,
                        workers=self.args.workers, sg=self.args.sg, seed=3778)

        # Build vocab over train samples
        self.model_w2v.build_vocab(token_review)

        # We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
        t0 = time()
        for _ in range(5):
            self.model_w2v.train(np.random.permutation(token_review), 
                            total_examples=self.model_w2v.corpus_count, 
                            epochs=self.model_w2v.epochs)
        elapsed=time() - t0
        if verbose: print(f'Time taken for Word2vec training: {elapsed} seconds.')


        # Save Word2Vec model
        # self.model_w2v.save(f'{W2V_DIR}w2v_model.model') # change this


        # List all words in embedding
        words_w2v = list(self.model_w2v.wv.vocab.keys())

        # Create dict for embedding matrix (word <-> row)
        self.row_dict = dict({word:self.model_w2v.wv.vocab[word].index for word in words_w2v})

        # Include Padding/Unknown indexes
        self.row_dict['_unknown_'] = len(words_w2v)
        self.row_dict['_padding_'] = len(words_w2v)+1

        # Define stopwords
        stopwords = STOP_WORDS.words('english')

        # Create word embedding matrix:
        self.embedding_matrix = np.zeros((len(words_w2v)+2, W2V_SIZE))

        for word in words_w2v:
            self.embedding_matrix[self.row_dict[word]][:] = np.array(self.model_w2v.wv[word])

            if args.reset_stopwords: # point stopwords to zeros
                if word in stopwords:
                    self.embedding_matrix[self.row_dict[word]][:] = np.zeros(W2V_SIZE)


        # Save embedding layer and row dict

        # with open(f'{W2V_DIR}MIMIC_emb_train_vec{W2V_SIZE}.pkl', 'wb') as file:
        #     pickle.dump(self.embedding_matrix, file)
            
        # with open(f'{W2V_DIR}MIMIC_dict_train_vec{W2V_SIZE}.pkl', 'wb') as file:
        #     pickle.dump(self.row_dict, file)

        if verbose:
            print(f'''
            W2V embedding matrix shape: {self.embedding_matrix.shape}
            ''')


    def transform(self, X): #(save=True?)

        # if not self.row_dict, load row_dict

        return (X
                .pipe(utils.preprocessor_word2vec)
                .apply(utils.convert_data_to_index, row_dict=self.row_dict)
                .apply(lambda x: np.squeeze(pad_sequences([x], padding = 'post', truncating = 'post',
                                            maxlen = MAX_LENGTH, value = self.row_dict['_padding_']))))

        # Save transformed embeddings? (as in MIMIC_process_inputs.py)

    def save(self):
        pass

    def load(self):
        pass
