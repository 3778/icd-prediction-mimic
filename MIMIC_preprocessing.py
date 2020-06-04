####
# MIMIC Preprocessing
# Merge NOTEEVENTS with DIAGNOSES_ICD tables
# This script creates a table with HADM_IDs linked to unique discharge summaries and a correspondent ICD codes
####


## Imports

import pandas as pd

from constants import *


## 1. Load MIMIC-III Tables

df_note = pd.read_csv(DATA_DIR + 'NOTEEVENTS.csv.gz')
df_icds = pd.read_csv(DATA_DIR + 'DIAGNOSES_ICD.csv.gz')


## 2. Preprocess NOTEEVENTS

# Select Discharge Summaries
df_text = df_note[df_note.CATEGORY == 'Discharge summary']

# Drop duplicates (duplicated notes)
df_text = df_text.drop_duplicates('TEXT')

# Keep only columns needed
# Keep only 1 document per HADM_ID
df_text = df_text[['SUBJECT_ID','HADM_ID','TEXT']].astype({'HADM_ID':'int32'}).drop_duplicates('HADM_ID')


## 3. Preprocess DIAGNOSES_ICD

# Drop NAN
df_icds.dropna(inplace=True)

# group by HADM_ID
id_group = df_icds.groupby(('HADM_ID'))

# Get ICDs as list
df_icds = id_group['ICD9_CODE'].unique().reset_index()


## 4. Merge both tables

df_mimic = pd.merge(df_icds,df_text,on='HADM_ID',how='inner')


## 5. Save dataset

pd.to_pickle(df_mimic, DATA_DIR + 'mimic3_data.pkl')


print('\n-------------\n')

print('Total unique ICD codes:', df_mimic.ICD9_CODE.explode().unique().shape[0])
print('Total samples:', df_mimic.shape[0])

print('\n\nData preprocessed! Now run MIMIC_train_w2v.py to train Word2Vec embeddings or MIMIC_train_baselines.py to train LR or Constant models.')


