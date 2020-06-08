### Train Word2Vec word vectors for the MIMIC-III dataset.

import argparse
import numpy as np
import pandas as pd
import pickle
import re
import random
from time import time
from gensim.models import Word2Vec
from nltk.corpus import stopwords as STOP_WORDS
import nltk
nltk.download('stopwords')

from constants import DATA_DIR, W2V_DIR, W2V_SIZE
import utils

def main(args):

    ### 1. Load MIMIC-III preprocessed data

    # Load DataFrame
    df = pd.read_pickle(DATA_DIR + 'mimic3_data.pkl')

    # Load splits
    train_ids = pd.read_csv(DATA_DIR + 'train' + '_full_hadm_ids.csv',header=None, names=['HADM_ID'])


    ### 2. Training Word2Vec

    # Get tokens of train set
    token_review = (df.query("HADM_ID.isin(@train_ids)")
                      ["TEXT"]
                      .pipe(utils.preprocessor))

    # Instantiate model
    model_w2v = Word2Vec(min_count=10, window=5, size=W2V_SIZE, sample=1e-3, negative=5, workers=args.workers, sg=args.sg, seed=3778)

    # Build vocab over train samples
    model_w2v.build_vocab(token_review)

    # We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
    Idx=list(range(len(token_review)))

    t0 = time()
    for epoch in range(5):
        random.shuffle(Idx)
        perm_sentences = [token_review[i] for i in Idx]
        model_w2v.train(perm_sentences,total_examples = model_w2v.corpus_count, epochs = model_w2v.epochs)
        #print(epoch)
    elapsed=time() - t0
    print(f'Time taken for Word2vec training: {elapsed} seconds.')


    # Save Word2Vec model
    model_w2v.save(W2V_DIR + 'w2v_model.model')

    ### 3. Create Embedding Layer

    # List all words in embedding
    words_w2v = list(model_w2v.wv.vocab.keys())

    # Create dict for embedding matrix (word <-> row)
    row_dict = dict({word:model_w2v.wv.vocab[word].index for word in words_w2v})

    # Include Padding/Unknown indexes
    row_dict['_unknown_'] = len(words_w2v)
    row_dict['_padding_'] = len(words_w2v)+1

    # Define stopwords
    stopwords = STOP_WORDS.words('english')

    # Create word embedding matrix:
    wv_embedding_matrix = np.zeros((len(words_w2v)+2, W2V_SIZE))

    for word in words_w2v:
        wv_embedding_matrix[row_dict[word]][:] = np.array(model_w2v.wv[word])

        if args.reset_stopwords: # point stopwords to zeros
            if word in stopwords:
                wv_embedding_matrix[row_dict[word]][:] = np.zeros(W2V_SIZE)

    # Save embedding layer and row dict

    matrix_str = W2V_DIR + 'MIMIC_emb_train' + '_vec' + str(W2V_SIZE) + '.pkl'
    dict_str = W2V_DIR + 'MIMIC_dict_train' + '_vec' + str(W2V_SIZE) + '.pkl'

    with open(matrix_str,'wb') as file:
        pickle.dump(wv_embedding_matrix, file)
        
    with open(dict_str,'wb') as file:
        pickle.dump(row_dict, file)

    print(f'''
    W2V embedding matrix shape: {wv_embedding_matrix.shape}
    Word2Vec embeddings saved! Now run MIMIC_process_inputs.py.
    ''')


## Parser
parser = argparse.ArgumentParser(description='Train Word2Vec word embeddings')
parser.add_argument('-workers', type=int, dest='workers', default=8, help='Number of CPU threads for W2V training.')
parser.add_argument('--reset_stopwords', type=bool, dest='reset_stopwords', default=0, help='True to set stopwords vectors to null. Default False.')
parser.add_argument('--train_method', type=bool, dest='sg', default=1, help='W2V train method. 0 for CBoW, 1 for Skipgram.')

args = parser.parse_args()

# Start 
main(args)
