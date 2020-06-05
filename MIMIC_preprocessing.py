####
# MIMIC Preprocessing
# Merge NOTEEVENTS with DIAGNOSES_ICD tables
# This script creates a table with HADM_IDs linked to unique discharge summaries and a correspondent ICD codes
####


## Imports

import pandas as pd

# Custom modules
from constants import *


df_text = (pd.read_csv(DATA_DIR + 'NOTEEVENTS.csv.gz')
             .query("CATEGORY == 'Discharge summary')
             .drop_duplicates('TEXT')
             .drop_duplicates('HADM_ID')
             [['SUBJECT_ID','HADM_ID','TEXT']])

df_icds = (pd.read_csv(DATA_DIR + 'DIAGNOSES_ICD.csv.gz')
             .dropna()
             .groupby('HADM_ID')
             ['ICD9_CODE']
             .unique()
             .reset_index())

df_mimic = pd.merge(df_icds,df_text,on='HADM_ID',how='inner')


## 5. Save dataset

pd.to_pickle(df_mimic, DATA_DIR + 'mimic3_data.pkl')


print('\n-------------\n')

print('Total unique ICD codes:', df_mimic.ICD9_CODE.explode().unique().shape[0])
print('Total samples:', df_mimic.shape[0])

print('\n\nData preprocessed! Now run MIMIC_train_w2v.py to train Word2Vec embeddings or MIMIC_train_baselines.py to train LR or Constant models.')

