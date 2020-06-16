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

def get_model(args):

    if args.MODEL_NAME == 'cte':
        return models.CTE_Model(args)

    elif args.MODEL_NAME == 'lr':
        return models.LR_Model(args)

    elif args.MODEL_NAME == 'cnn':
        return models.CNN_Model(args)

    elif args.MODEL_NAME == 'gru':
        return models.GRU_Model(args)

    elif args.MODEL_NAME == 'cnn_att':
        return models.CNNAtt_Model(args)