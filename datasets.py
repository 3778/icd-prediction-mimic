import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MultiLabelBinarizer

from constants import DATA_DIR

import utils



class MIMIC_Dataset:

    def __init__(self):
        self.name = 'MIMIC'

    def load_preprocessed(self, path=DATA_DIR):
        with open(f'{path}mimic3_data.pkl', 'rb') as file:
            self.df = pickle.load(file) 

    def save_preprocessed(self, path=DATA_DIR):
        pd.to_pickle(self.df, f'{path}mimic3_data.pkl') ######### SAMPLING FOR TEST

    def preprocess(self, verbose=1):

        df_text = (pd.read_csv(f'{DATA_DIR}NOTEEVENTS.csv.gz')
                .query("CATEGORY == 'Discharge summary'")
                .drop_duplicates('TEXT')
                .drop_duplicates('HADM_ID')
                [['SUBJECT_ID','HADM_ID','TEXT']])

        df_icds = (pd.read_csv(f'{DATA_DIR}DIAGNOSES_ICD.csv.gz')
                    .dropna()
                    .groupby('HADM_ID')
                    ['ICD9_CODE']
                    .unique()
                    .reset_index())

        self.df = pd.merge(df_icds, df_text, on='HADM_ID', how='inner')
        

        if verbose:
            print(f'''
            -------------
            Total unique ICD codes: {self.df.ICD9_CODE.explode().nunique()}
            Total samples: {self.df.shape[0]}
            Data preprocessed!
            ''')

    def split(self, hadm_ids=None, verbose=1):

        # Fit multi-hot encoder
        hist = utils.make_icds_histogram(self.df)
        self.all_icds = hist.index.tolist()
        self.mlb = MultiLabelBinarizer(self.all_icds).fit(self.df['ICD9_CODE'])

        if not hadm_ids: # change this
            train_ids = utils.load_ids_from_txt(f'{DATA_DIR}train_full_hadm_ids.csv')
            val_ids = utils.load_ids_from_txt(f'{DATA_DIR}dev_full_hadm_ids.csv')
            test_ids = utils.load_ids_from_txt(f'{DATA_DIR}test_full_hadm_ids.csv')

            hadm_ids = [train_ids, val_ids, test_ids]
    
        assert not np.in1d(hadm_ids[0], hadm_ids[1]).any(), 'Data leakage!'
        assert not np.in1d(hadm_ids[0], hadm_ids[2]).any(), 'Data leakage!'
        assert not np.in1d(hadm_ids[2], hadm_ids[1]).any(), 'Data leakage!'

        self.all_icds_train = utils.make_icds_histogram(self.df.query("HADM_ID.isin(@hadm_ids[0])")).index.tolist()

        self.x_train = self.df.query("HADM_ID.isin(@hadm_ids[0])").TEXT
        self.y_train = self.mlb.transform(self.df.query("HADM_ID.isin(@hadm_ids[0])").ICD9_CODE).tolist()

        self.x_val = self.df.query("HADM_ID.isin(@hadm_ids[1])").TEXT
        self.y_val = self.mlb.transform(self.df.query("HADM_ID.isin(@hadm_ids[1])").ICD9_CODE).tolist()

        self.x_test = self.df.query("HADM_ID.isin(@hadm_ids[2])").TEXT
        self.y_test = self.mlb.transform(self.df.query("HADM_ID.isin(@hadm_ids[2])").ICD9_CODE).tolist()
        
        if verbose:
            print(f'''
            Data Split: {self.x_train.shape[0]}, {self.x_val.shape[0]}, {self.x_test.shape[0]}
            ''')







    

    

    








