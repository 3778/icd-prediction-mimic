import numpy as np
import pickle
import pandas as pd

from constants import SAVE_DIR, W2V_DIR, W2V_SIZE, MAX_LENGTH
import model_functions as fun
import models

def make_icds_histogram(df):
    return df.ICD9_CODE.explode().value_counts()


def load_ids_from_txt(filepath):
    with open(filepath, 'r') as f:
        return f.read().split()


def preprocessor(text_series):
    return (text_series
            .str.replace('<[^>]*>', '')
            .str.lower()
            .str.replace('[\W]+', ' ')
            .str.split())


def preprocessor_tfidf(text_series):
    return (text_series
            .str.replace('\[\*\*[^\]]*\*\*\]','')
            .str.replace('<[^>]*>', '')
            .str.replace('[\W]+', ' ')
            .str.lower()
            .str.replace(' \d+', ' '))


def preprocessor_word2vec(text_series):
    return (text_series
            .str.replace('\[\*\*[^\]]*\*\*\]','')
            .str.replace('<[^>]*>', '')
            .str.replace('[\W]+', ' ')
            .str.lower()
            .str.replace(' \d+', ' ')
            .str.split())


def convert_data_to_index(string_data, row_dict):
    return [row_dict.get(word, row_dict['_unknown_']) for word in string_data]


def split(df, mlb, all_icds, train_ids, val_ids, test_ids, to_array=True):

    assert not np.in1d(train_ids, val_ids).any()
    assert not np.in1d(train_ids, test_ids).any()
    assert not np.in1d(test_ids, val_ids).any()
       
    # Split by HADM_IDS
    train_set = df.query("HADM_ID.isin(@train_ids)")
    val_set = df.query("HADM_ID.isin(@val_ids)")
    test_set = df.query("HADM_ID.isin(@test_ids)")

    print(f'''
    Data Split:{train_set.shape[0]}, {val_set.shape[0]}, {test_set.shape[0]}
    ''')
    
    x = []
    y = []
    for subset in [train_set, val_set, test_set]:
        if to_array:
            x.append(np.vstack(subset['int_seq'].to_list())) # Change this
        else: 
            x.append(subset['clean_text'].to_list())

        y.append(mlb.transform(subset['ICD9_CODE']))
        
    # Call model_args class
    model_args = fun.model_args(x,y)
    
    return model_args


# def get_model(model_args, embedding_matrix = None, args = None):

#     if args.MODEL_NAME == 'lr':
#         return models.lr_model(model_args, args)

#     elif args.MODEL_NAME == 'cnn':
#         return models.cnn_model(model_args, embedding_matrix, args)

#     elif args.MODEL_NAME == 'gru':
#         return models.gru_model(model_args, embedding_matrix, args)

#     elif args.MODEL_NAME == 'cnn_att':
#         return models.cnn_att_model(model_args, embedding_matrix, args)


def load_model_custom(model_args, embedding_matrix=None, args=None):

    if args == None:
        with open(f'{SAVE_DIR}args.pkl','rb') as file:
            args = pickle.load(file)

    else:
        model = get_model(model_args=model_args,embedding_matrix=embedding_matrix,args=args)
        model.load_weights(SAVE_DIR + args.MODEL_NAME + '.h5')

        return model

def load_w2v_emb(w2v_vec_size=W2V_SIZE, dataset='MIMIC', verbose=0):
    """
    Function to load trained w2v embedding matrix and correspondent row dictionary.

    Parameters
    -----
    w2v_vec_size: vector dimension used for Word2Vec training.
    dataset: dataset where samples come from.
    add_descriptions (optional): additional descriptions used when saving x, y and embedding matrix.
    
    Outputs
    -----
    W2V embedding matrix and dictionary to link tokens to matrix index.
    """

    # Load embedding matrix
    with open(f'{W2V_DIR}{dataset}_emb_train_vec{w2v_vec_size}.pkl','rb') as file:
        w2v_embedding_matrix = pickle.load(file)
    
    # Load row_dict
    with open(f'{W2V_DIR}{dataset}_dict_train_vec{w2v_vec_size}.pkl','rb') as file:
        w2v_row_dict = pickle.load(file)

    if verbose:
        print(f'''
        Dataset: {dataset}
        Embedding matrix shape: {w2v_embedding_matrix.shape[0]}
        ''')

    return w2v_embedding_matrix, w2v_row_dict

    
def load_w2v_proc_inputs(max_words=MAX_LENGTH, dataset='MIMIC', verbose=0):
    """
    Function to load processed x (truncated, padded, converted to indexes for a given embedding matrix) and y (converted to multi-hot encoding of classes) data.

    Parameters
    -----
    max_words: fixed-length input of preprocessed samples.
    dataset: dataset where samples come from.
    add_descriptions (optional): additional descriptions used when saving x, y and embedding matrix.

    Outputs
    -----
    Preprocessed x and y, and MultilabelBinarizer instance to allow multi-hot<->ICDs conversion.
    """

    # Load x
    with open(f'{W2V_DIR}{dataset}_x_pad{max_words}.pkl','rb') as file:
        x = pickle.load(file)
    
    # Load y
    with open(f'{W2V_DIR}{dataset}_y.pkl','rb') as file:
        y = pickle.load(file)

    # Load mlb
    with open(f'{W2V_DIR}{dataset}_mlb.pkl','rb') as file:
        mlb = pickle.load(file)

    if verbose:
        print(f"""
        Train set: X: {x[0].shape}, Y: {y[0].shape}
        Val set: X: {x[1].shape}, Y: {y[1].shape}
        Test set: X: {x[2].shape}, Y: {y[2].shape}
        """)

    return x, y, mlb


################################

def get_model(args):

    if args.MODEL_NAME == 'lr':
        return models.LR_Model(args)

    elif args.MODEL_NAME == 'cnn':
        return models.CNN_Model(args)

    elif args.MODEL_NAME == 'gru':
        return models.GRU_Model(args)

    elif args.MODEL_NAME == 'cnn_att':
        return models.CNNAtt_Model(args)