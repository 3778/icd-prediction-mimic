##
# Training Baseline Models LR and Constant
#
##




#######################
# save MLB
# maybe the fit calls could go in a separate function in model_functions (as make_model)
#######################

## Imports
import argparse
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import TensorBoard


import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords as STOP_WORDS

# Custom modules
from constants import *
import model_functions as fun
import utils

def main(args):

    # Load DataFrame
    df = pd.read_pickle(DATA_DIR + 'mimic3_data.pkl')

    # Get ICD list
    hist = utils.make_icds_histogram(df)
    all_icds = hist.index.tolist()

    # Load splits
    train_ids = pd.read_csv(DATA_DIR + 'train'+'_full_hadm_ids.csv',header=None, names=['HADM_ID'])
    val_ids = pd.read_csv(DATA_DIR + 'dev'+'_full_hadm_ids.csv',header=None, names=['HADM_ID']) 
    test_ids = pd.read_csv(DATA_DIR + 'test'+'_full_hadm_ids.csv',header=None,names=['HADM_ID']) 

    # Fit multi-hot encoder
    mlb = MultiLabelBinarizer(all_icds)
    mlb.fit(df['ICD9_CODE'])

    # Apply preprocessing to free text
    df['clean_text'] = df['TEXT'].apply(utils.preprocessor_tfidf)

    # Split
    model_args = utils.split_LR(df, mlb, all_icds, train_ids, val_ids, test_ids)

    if args.MODEL_NAME=='lr':

        ## 1. Compute TF-IDF features

        # Instantiate TF-IDF Transformer
        tfidf = TfidfVectorizer(stop_words = STOP_WORDS.words('english'), max_features=args.max_features)

        # Fit for train set
        model_args.x[0] = tfidf.fit_transform(model_args.x[0])

        # Transform other sets
        model_args.x[1] = tfidf.transform(model_args.x[1])
        model_args.x[2] = tfidf.transform(model_args.x[2])

        ## 2. Fit model (this could be a separate function)

        tf.keras.backend.clear_session()

        # Instantiate callbacks
        tensorboard_callback = TensorBoard(log_dir = SAVE_DIR + 'logs/fit/' + args.MODEL_NAME)
        f1_callback = fun.f1_callback_save_weights(model_args, tb_callback = tensorboard_callback, best_name= SAVE_DIR + args.MODEL_NAME + '.h5')

        callbacks = [tensorboard_callback, f1_callback]

        # Call model
        model_args.model = utils.get_model(model_args=model_args, args=args)

        if args.verbose: model_args.model.summary()

        # Fit
        history = model_args.model.fit(model_args.x[0], model_args.y[0], validation_data=(model_args.x[1],model_args.y[1]),
                                epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=args.verbose)

        # Restore weights from the best epoch based on F1 val with optimized threshold
        best_model = utils.get_model(model_args=model_args, args=args)
        best_model.load_weights(SAVE_DIR + args.MODEL_NAME + '.h5')

        # Predict
        model_args.predict(custom_model = best_model)

        # Compute metrics @ best threshold
        print('\n--------------------\nMetrics @ %.2f for best epoch:\n' %model_args.best_t)
        model_args.metrics(threshold = model_args.best_t)
        print('\n--------------------\n')

        # Save args to correctly load weights
        with open(SAVE_DIR + MODEL_NAME + '_args.pkl', 'wb') as file:
            pickle.dump(args, file)


    else:

        f1_val = []

        # k fixed-length size of prediction outputs for each sample
        ks = np.linspace(1,20,20, dtype=int)
        
        # Select most occuring ICDs in train set
        most_occ_train = utils.make_icds_histogram(df[df.HADM_ID.isin(train_ids.HADM_ID)]).index.tolist()


        # Sweep k for val set
        for k in ks:
            # Repeat same prediction for every sample
            y_pred = np.repeat([most_occ_train[:k]],repeats=df.shape[0],axis=0)     
            # Binarize y_pred
            y_pred = mlb.transform(y_pred)

            # Repeat for Val set
            model_args.y_pred[1] = y_pred[:model_args.y[1].shape[0]]

            # Compute metrics
            print('\nMetrics when predicting the %d most occurring ICD codes:' %k)
            model_args.metrics([0,1,0])

            # Store F1 val
            f1_val.append(model_args.f1_score[1])
        
        best_k = ks[np.argmax(np.ravel(f1_val))]
        print('\n\nBest F1 val micro score =',max(f1_val),'[',best_k,'ICDs predicted ]')

        # Recompute predictions of best value

        # Repeat same prediction for every sample
        y_pred = np.repeat([most_occ_train[:best_k]],repeats=df.shape[0],axis=0)     
        # Binarize y_pred
        y_pred = mlb.transform(y_pred)

        # Repeat same prediction for each subset
        model_args.y_pred[0] = y_pred[:model_args.y[0].shape[0]]
        model_args.y_pred[1] = y_pred[:model_args.y[1].shape[0]]
        model_args.y_pred[2] = y_pred[:model_args.y[2].shape[0]]

        # Compute metrics
        print('\n-----------------\nMetrics when predicting the %d most occurring ICD codes:\n' %best_k)
        model_args.metrics()







        
## Parser
parser = argparse.ArgumentParser(description='Train model for MIMIC-III dataset and compute metrics.')
parser.add_argument('-model', type=str, dest='MODEL_NAME', choices=['lr', 'cte'], default = 'lr',help='Model for training.')
parser.add_argument('-epochs', type=int, dest='epochs', default=10, help='Number of epochs.')
parser.add_argument('-tfidf_maxfeatures', type=int, dest='max_features', default=20000, help='Max features for TF-IDF.')
parser.add_argument('-batch_size', type=int, dest='batch_size', default=32, help='Batch Size.')
parser.add_argument('-lr', type=float, dest='lr', default=0, help='Learning Rate. 0 for article optimized value.')
parser.add_argument('--verbose', type=int, dest='verbose', default=2, help='Verbose when training.')

args = parser.parse_args()


main(args)