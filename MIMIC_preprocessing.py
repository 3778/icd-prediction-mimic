####
# MIMIC Preprocessing
# Merge NOTEEVENTS with DIAGNOSES_ICD tables
# This script creates a table with HADM_IDs linked to unique discharge summaries and a correspondent ICD codes
####

import pandas as pd

from constants import DATA_DIR


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

pd.to_pickle(df_mimic, DATA_DIR + 'mimic3_data.pkl')


print(f'''
-------------
Total unique ICD codes: {df_mimic.ICD9_CODE.explode().nunique()}
Total samples: {df_mimic.shape[0]}
Data preprocessed! Now run MIMIC_train_w2v.py to train Word2Vec embeddings or MIMIC_train_baselines.py to train LR or Constant models.
''')
