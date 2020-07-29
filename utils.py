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
import pickle
import pandas as pd
from tensorflow.keras.callbacks import LearningRateScheduler

from constants import SAVE_DIR, W2V_DIR, W2V_SIZE, MAX_LENGTH
import model_functions as fun
import models


def make_icds_histogram(df):
    return df.ICD9_CODE.explode().value_counts()


def load_list_from_txt(filepath):
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


def lr_schedule_callback(args):
    # Create scheduler function
    def scheduler(epoch):
        if epoch < args.epoch_drop:
            return args.initial_lr   
        else:
            return args.final_lr
    
    return LearningRateScheduler(scheduler, verbose=1)


def get_model(args=None, load_path=None):

    if args.MODEL_NAME == 'cte':
        return models.CTE_Model(args, load_path)

    elif args.MODEL_NAME == 'lr':
        return models.LR_Model(args, load_path)

    elif args.MODEL_NAME == 'cnn':
        return models.CNN_Model(args, load_path)

    elif args.MODEL_NAME == 'gru':
        return models.GRU_Model(args, load_path)

    elif args.MODEL_NAME == 'cnn_att':
        return models.CNNAtt_Model(args, load_path)