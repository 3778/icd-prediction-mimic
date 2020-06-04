##
#
# Utils for repo
#
##


####################
# fix get_model
#
#####################

import numpy as np
import pickle
import pandas as pd
import re

import model_functions as fun
import models


from constants import *



def make_icds_histogram(df):
    return df.ICD9_CODE.explode().value_counts()


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = text.split()
    return text


def preprocessor_tfidf(text):
    text = re.sub('\[\*\*[^\]]*\*\*\]', '', text)
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower()) 
    text = re.sub(" \d+", " ", text)
    return text


def preprocessor_word2vec(text):
    text = re.sub('\[\*\*[^\]]*\*\*\]', '', text)
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower()) 
    text = re.sub(" \d+", " ", text)
    text = text.split()
    return text


def convert_data_to_index(string_data, row_dict):
    index_data = []
    for word in string_data:
        if word in row_dict.keys():
            index_data.append(row_dict[word])
        else:
            index_data.append(row_dict['_unknown_'])
    return index_data


def split(df, mlb, all_icds, train_ids, val_ids, test_ids):
       
    # Split by HADM_IDS
    train_set = df[df.HADM_ID.isin(train_ids.HADM_ID)]
    val_set = df[df.HADM_ID.isin(val_ids.HADM_ID)]
    test_set = df[df.HADM_ID.isin(test_ids.HADM_ID)]
    
    print('\nData Split:',train_set.shape[0], val_set.shape[0],test_set.shape[0],'\n')
    
    # Call model_args class
    model_args = fun.model_args()
    
    # Train
    model_args.x[0] = np.vstack(train_set['int_seq'].to_list())
    model_args.y[0] = mlb.transform(train_set['ICD9_CODE'])

    # Val
    model_args.x[1] = np.vstack(val_set['int_seq'].to_list())
    model_args.y[1] = mlb.transform(val_set['ICD9_CODE'])

    #Test
    model_args.x[2] = np.vstack(test_set['int_seq'].to_list())
    model_args.y[2] = mlb.transform(test_set['ICD9_CODE'])
    
    return model_args

def split_LR(df, mlb, all_icds, train_ids, val_ids, test_ids):
    
   
    # Split by HADM_IDS
    train_set = df[df.HADM_ID.isin(train_ids.HADM_ID)]
    val_set = df[df.HADM_ID.isin(val_ids.HADM_ID)]
    test_set = df[df.HADM_ID.isin(test_ids.HADM_ID)]
    
    print('\nData Split:',train_set.shape[0], val_set.shape[0],test_set.shape[0],'\n')
    
    # Call model_args class
    model_args = fun.model_args()
    
    # Train
    model_args.x[0] = train_set['clean_text'].to_list()
    model_args.y[0] = mlb.transform(train_set['ICD9_CODE'])

    # Val
    model_args.x[1] = val_set['clean_text'].to_list()
    model_args.y[1] = mlb.transform(val_set['ICD9_CODE'])

    #Test
    model_args.x[2] = test_set['clean_text'].to_list()
    model_args.y[2] = mlb.transform(test_set['ICD9_CODE'])
    
    return model_args


def get_model(model_args, embedding_matrix = None, args = None):

    if args.MODEL_NAME == 'lr':
        return models.lr_model(model_args,args)

    elif args.MODEL_NAME == 'cnn':
        return models.cnn_model(model_args, embedding_matrix, args)

    elif args.MODEL_NAME == 'gru':
        return models.gru_model(model_args, embedding_matrix, args)

    elif args.MODEL_NAME == 'cnn_att':
        return models.cnn_att_model(model_args, embedding_matrix, args)


def load_model_custom(model_args, embedding_matrix=None, args=None):

    if args == None:
        with open(SAVE_DIR + 'args.pkl','rb') as file:
            args = pickle.load(file)

    else:
        model = get_model(model_args=model_args,embedding_matrix=embedding_matrix,args=args)
        model.load_weights(SAVE_DIR + args.MODEL_NAME + '.h5')

        return model

def load_w2v_emb(w2v_vec_size=W2V_SIZE, dataset='MIMIC', add_descriptions='', verbose=0):
    """
    Function to load trained w2v embedding matrix and correspondent row dictionary.

    Parameters
    -----
    w2v_vec_size: vector dimension used for Word2Vec training.\n
    dataset: dataset where samples come from.\n
    add_descriptions (optional): additional descriptions used when saving x, y and embedding matrix.
    
    Outputs
    -----
    W2V embedding matrix and dictionary to link tokens to matrix index.\n
    """

    # Load embedding matrix
    with open(W2V_DIR + dataset + '_emb_train_vec' + str(w2v_vec_size) + str(add_descriptions) + '.pkl','rb') as file:
        w2v_embedding_matrix = pickle.load(file)
    
    # Load row_dict
    with open(W2V_DIR + dataset + '_dict_train_vec' + str(w2v_vec_size) + str(add_descriptions) + '.pkl','rb') as file:
        w2v_row_dict = pickle.load(file)

    if verbose:
        print('Dataset:', dataset)
        print('Embedding matrix shape:', w2v_embedding_matrix.shape)

    return w2v_embedding_matrix, w2v_row_dict

    
def load_w2v_proc_inputs(max_words=MAX_LENGTH, dataset='MIMIC', add_descriptions='', verbose=0):
    """
    Function to load processed x (truncated, padded, converted to indexes for a given embedding matrix) and y (converted to multi-hot encoding of classes) data.

    Parameters
    -----
    max_words: fixed-length input of preprocessed samples.\n
    dataset: dataset where samples come from.\n
    add_descriptions (optional): additional descriptions used when saving x, y and embedding matrix.

    Outputs
    -----
    Preprocessed x and y, and MultilabelBinarizer instance to allow multi-hot<->ICDs conversion.\n
    """

    # Load x
    with open(W2V_DIR + dataset + '_x' + '_pad' + str(max_words) + str(add_descriptions) + '.pkl','rb') as file:
        x = pickle.load(file)
    
    # Load y
    with open(W2V_DIR + dataset + '_y' + '.pkl','rb') as file:
        y = pickle.load(file)

    # Load mlb
    with open(W2V_DIR + dataset + '_mlb' + '.pkl','rb') as file:
        mlb = pickle.load(file)

    if verbose:
        print('Train set: X:', x[0].shape, 'Y:', y[0].shape)
        print('Val set: X:', x[1].shape, 'Y:', y[1].shape)
        print('Test set: X:', x[2].shape, 'Y:', y[2].shape)

    return x, y, mlb




