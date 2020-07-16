## Process inputs based on embedding matrix

import numpy as np
import pandas as pd
import pickle
import re
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from constants import DATA_DIR, W2V_DIR, MAX_LENGTH
import utils


## 1. Load MIMIC-III preprocessed data

# Load DataFrame
df = pd.read_pickle(f'{DATA_DIR}mimic3_data.pkl')

# Get ICD list
hist = utils.make_icds_histogram(df)
all_icds = hist.index.tolist()

# Load splits
train_ids = utils.load_ids_from_txt(f'{DATA_DIR}train_full_hadm_ids.csv')
val_ids = utils.load_ids_from_txt(f'{DATA_DIR}dev_full_hadm_ids.csv')
test_ids = utils.load_ids_from_txt(f'{DATA_DIR}test_full_hadm_ids.csv')

## 2. Load Embedding 

wv_embedding_matrix, row_dict = utils.load_w2v_emb()


## 3. Preprocess, pad and truncate

# Apply preprocessing
df['int_seq'] = (df['TEXT']
                .pipe(utils.preprocessor_word2vec)
                .apply(utils.convert_data_to_index, row_dict=row_dict)
                .apply(lambda x: np.squeeze(pad_sequences([x], padding = 'post', truncating = 'post',
                                            maxlen = MAX_LENGTH, value = row_dict['_padding_']))))


## 3. Split data

# Multi-label binarizer (multi-hot encode labels)
mlb = MultiLabelBinarizer(all_icds).fit(df['ICD9_CODE'])

model_args = utils.split(df, mlb, all_icds, train_ids, val_ids, test_ids)


## 4. Save processed and splited data

with open(f'{W2V_DIR}MIMIC_x_pad{MAX_LENGTH}.pkl','wb') as file:
    pickle.dump(model_args.x, file)

with open(f'{W2V_DIR}MIMIC_y.pkl','wb') as file:
    pickle.dump(model_args.y, file)

with open(f'{W2V_DIR}MIMIC_mlb.pkl','wb') as file:
    pickle.dump(mlb, file)


print('''
    Data processed! Now go ahead and run MIMIC_train_nn.py to train any NN model.
    ''')
