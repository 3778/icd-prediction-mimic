#####
# Process inputs based on embedding matrix
#
#####


######################
# 
# 
######################

## Imports

# Global variables
from constants import *

# Modules
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Custom Modules
import utils


## 1. Load MIMIC-III preprocessed data

# Load DataFrame
df = pd.read_pickle(DATA_DIR + 'mimic3_data.pkl')

# Get ICD list
hist = utils.make_icds_histogram(df)
all_icds = hist.index.tolist()

# Load splits
train_ids = pd.read_csv(DATA_DIR + 'train'+'_full_hadm_ids.csv',header=None, names=['HADM_ID'])
val_ids = pd.read_csv(DATA_DIR + 'dev'+'_full_hadm_ids.csv',header=None, names=['HADM_ID']) 
test_ids = pd.read_csv(DATA_DIR + 'test'+'_full_hadm_ids.csv',header=None,names=['HADM_ID']) 


## 2. Load Embedding 

wv_embedding_matrix, row_dict = utils.load_w2v_emb()


## 3. Padd and truncate

# Set padding / truncation 
padding = 'post'
truncating = 'post'

# Apply preprocessing
df['clean_text'] = df['TEXT'].apply(utils.preprocessor_word2vec)

# Convert words to correspondent rows in embedding matrix
df['int_seq'] = df['clean_text'].apply(lambda x: utils.convert_data_to_index(x, row_dict))

# Pad/truncate to max_words length
df['int_seq'] = df['int_seq'].apply(lambda x: np.squeeze(pad_sequences([x], padding = padding, truncating = truncating,maxlen = MAX_LENGTH, value = row_dict['_padding_'])))


## 3. Split data

# Multi-label binarizer (multi-hot encode labels)
mlb = MultiLabelBinarizer(all_icds)
mlb.fit(df['ICD9_CODE'])

model_args = utils.split(df, mlb, all_icds, train_ids, val_ids, test_ids)


## 4. Save processed and splited data

x_str = W2V_DIR + 'MIMIC' + '_x' + '_pad' + str(MAX_LENGTH) + str('') + '.pkl'
y_str = W2V_DIR + 'MIMIC' + '_y' + '.pkl'
mlb_str = W2V_DIR + 'MIMIC' + '_mlb' + '.pkl' 

with open(x_str,'wb') as file:
    pickle.dump(model_args.x, file)

with open(y_str,'wb') as file:
    pickle.dump(model_args.y, file)

with open(mlb_str,'wb') as file:
    pickle.dump(mlb, file)


print('\n Data processed! Now go ahead and run MIMIC_train_nn.py to train any NN model.\n')