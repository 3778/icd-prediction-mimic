import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model

import utils

## Class for model prediction and metrics
class model_args:
    """
    Class to store inputs, outputs and the tf.keras model, and compute predicitons and metrics.

    Parameters
    -----
    x:[x_train, x_val, x_test]\n
    y:[y_train, y_val, y_test]\n
    model: A tf.keras model object.\n
    """

    def __init__(self, x=[None]*3, y=[None]*3):
        self.x = x
        self.y = y
        self.y_pred = [None]*3
        self.model = None
        self.y_sparse = None

    
    def predict(self, train_val_test = [1,1,1], custom_model=None):
        """ 
        Get model predictions.
        
        Parameters
        -----
        train_val_test: Choose which subsets to evaluate metrics [train, validation, test].\n
        custom_model: Input another trained model. Must be a fitted model object.

        Outputs
        -----
        Outputs are stored in:\n
        self.y_pred[train,val,test]\n
        """
        for i in range(3):
            if train_val_test[i]:
                if custom_model:
                    self.y_pred[i] = custom_model.predict(self.x[i])
                else:
                    self.y_pred[i] = self.model.predict(self.x[i])

    
    def metrics(self, train_val_test = [1,1,1], avg='micro', threshold = 0.5, k=None, verbose=1):
        """ 
        Compute metrics after prediciton.
        
        Parameters
        -----
        train_val_test: Choose which subsets to evaluate metrics [train, validation, test].\n        
        avg: average method to compute metrics in a multi-label problem.\n        
        threshold: if k=None, sets the threshold for binarizing model outputs.\n        
        k: if not None, sets the fixed length k of the outputs. The k highest scores from the model will be the ICD outputs.

        Outputs
        -----
        Outputs are stored in:\n
        self.f1_score[train, val, test]
        self.precision[train, val, test]
        self.recall[train, val, test]\n
        """
        self.f1_score = np.zeros(3)
        self.precision = np.zeros(3)
        self.recall = np.zeros(3)

        # Get sparse representation (faster)
        if self.y_sparse == None:
            self.y_sparse = [csr_matrix(self.y[i]) for i in range(3)]

        def print_res(prec, rec, f1, subset, avg):
            print('--%s %s-- metrics:' %(subset, avg))
            print('F1\t\t\tPrecision\t\tRecall')
            print(f1,'\t', prec, '\t',rec, sep='')
        
        subset = ['Train','Val','Test']

        y_pred_bin = [None]*3
        for i in range(3):

            # From the chosen subsets:
            if train_val_test[i]:
                y_pred_bin[i] = np.zeros(self.y_pred[i].shape)

                # If approach is fixed-k output
                if k:
                    prob_rank = np.argsort(np.array(self.y_pred[i]))

                    for sample in range(len(y_pred_bin[i])):
                        y_pred_bin[i][sample][prob_rank[sample][-k:]] = 1

                # Else if approach is threshold-based
                else:
                    y_pred_bin[i]= self.y_pred[i] > threshold

                # Compute metrics
                self.precision[i], self.recall[i], self.f1_score[i], _ = precision_recall_fscore_support(self.y_sparse[i], csr_matrix(y_pred_bin[i]), average=avg,zero_division=0)

                # Print metrics
                if verbose: print_res(self.precision[i], self.recall[i], self.f1_score[i], subset[i], avg)



def sweep_thresholds(model_args, train_val_test=[1,1,1], thresholds = np.linspace(0.01,0.5,50), avg='micro', custom_model=None, predicted_model=False):
    """
    Function to sweep across desired thresholds, computing model metrics @ threshold in the validation set.
    The best threshold based on F1 is stored and metrics @ best_threshold are displayed for all subsets.

    Parameters
    -----
    model_args: class model args with x, y and fitted model.\n
    trai_val_test: choose which subsets to predict and compute. \n
    thresholds: list of thresholds to compute metrics.\n
    avg: average method for multi-label metrics computation.\n
    custom_model: optional, input a custom tf.keras fitted model.\n
    predicted_model: optional, inform that model predictions were already computed (and stored in model_args.y_pred).

    Ouputs
    -----
    model_args, with:
    - model_args.metrics with computed metrics @ best threshold
    - model_args.sweep results: {'thresholds': thresholds used,
                                'f1_val': f1 val per threshold,
                                'prec_val': precision val per threshold,
                                'rec_val': recall val per threshold,
                                'avg_val_pred': average number of ICDs predicted per sample
                                } \n
    """

    # Predict
    if not predicted_model:
        assert train_val_test[1]==1, 'If model is not predicted, train_val_set[1] must be 1. '
        if custom_model:
            model_args.predict(train_val_test=train_val_test,custom_model=custom_model)
        else:
            model_args.predict(train_val_test=train_val_test)

    sweep_f1_val = []
    sweep_prec_val = []
    sweep_rec_val = []
    sweep_avg_val_pred = []

    # Sweep through thresholds
    for thresh in thresholds:

        model_args.metrics([0,1,0], threshold=thresh, avg=avg, verbose=0)
        sweep_f1_val.append(model_args.f1_score[1])
        sweep_prec_val.append(model_args.precision[1])
        sweep_rec_val.append(model_args.recall[1])
        

        sweep_avg_val_pred.append(np.mean(sum((model_args.y_pred[1]>thresh).T)))

    best_t = thresholds[np.argmax(sweep_f1_val)]

    print('Best thresh: %.2f' %best_t)
    print('F1 @ %.2f:' %best_t,'\n')
    model_args.metrics(train_val_test=train_val_test,threshold=best_t, avg=avg)

    model_args.sweep_results = {'thresholds': thresholds,
                                'f1_val': sweep_f1_val,
                                'prec_val': sweep_prec_val,
                                'rec_val': sweep_rec_val,
                                'avg_val_pred': sweep_avg_val_pred,
                                'best_f1_val': model_args.f1_score[1],
                                'best_threshold': best_t,
                                }

    return model_args

def sweep_k(model_args, ks = np.linspace(1,20,20, dtype='int32'), avg='micro', custom_model=None, predicted_model=False):
    """
    Function to sweep across desired k (length of output fixed list), computing model metrics @ k in the validation set.
    The best k based on F1 is stored and metrics @ best_k are displayed for all subsets.

    Parameters
    -----
    model_args: class model args with x, y and fitted model.\n
    ks: list of k values to evaluate metrics on.\n
    avg: average method for multi-label metrics computation.\n
    custom_model: optional, input a custom tf.keras fitted model.\n
    predicted_model: optional, inform that model predictions were already computed (and stored in model_args.y_pred).

    Outputs
    -----
    model_args, with:
    - model_args.metrics with computed metrics @ best k
    - model_args.sweep results: {'ks': k values used,
                                'f1_val': f1 val per k,
                                'prec_val': precision val per k,
                                'rec_val': recall val per k,
                                } \n
    """
    # Predict
    if not predicted_model:
        if custom_model:
            model_args.predict(custom_model=custom_model)
        else:
            model_args.predict()

    sweep_f1_val = []
    sweep_prec_val = []
    sweep_rec_val = []

    # Sweep through k lengths
    for k in ks:

        model_args.metrics([0,1,0], k=k, avg=avg, verbose=0)
        sweep_f1_val.append(model_args.f1_score[1])
        sweep_prec_val.append(model_args.precision[1])
        sweep_rec_val.append(model_args.recall[1])


    best_k = ks[np.argmax(sweep_f1_val)]

    print('Best k: %d' %best_k)
    print('F1 @ %d:' %best_k,'\n')
    model_args.metrics(k=best_k, avg=avg)

    model_args.sweepk_results = {'ks': ks,
                                'f1_val': sweep_f1_val,
                                'prec_val': sweep_prec_val,
                                'rec_val': sweep_rec_val
                                }

    return model_args


class f1_callback_save_weights(Callback):
    """
    Callback for use in keras. 
    Computes f1 in validation set (model_args.x[1], model_args.y[1]) and prints it.
    Optionally, it can save the best model based on f1_val.
    If a tensorboard callback is specified, f1_val is stored in its logs.
    """
    
    def __init__(self, model_args, tb_callback=None, store_best=True, best_name='best_model.h5', avg='micro'):
        self.model_args = model_args
        self.tb_callback = tb_callback
        self.avg = avg
        self.best_f1_val = 0
        self.store_best = store_best
        self.best_name = best_name

    # Execute after each epoch    
    def on_epoch_end(self, epoch, logs={}):

        # Predict and compute f1 val
        self.model_args.predict([0,1,0], custom_model=self.model)   
        self.model_args.metrics([0,1,0], avg=self.avg, verbose=1)
        self.model_args = sweep_thresholds(self.model_args,train_val_test=[0,1,0], predicted_model=True)
        print('')
        
        # Set items to store in Tensorboard
        items_to_write={
            "f1_val": self.model_args.f1_score[1]
        }
        
        # Send to Tensorboard logs
        if self.tb_callback != None:
            writer = self.tb_callback.writer
            for name, value in items_to_write.items():
                summary = tf.summary.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = name
                writer.add_summary(summary, epoch)
                writer.flush()  

        # Store best model
        if self.store_best:
            if self.model_args.f1_score[1] > self.best_f1_val:
                self.best_model = self.model_args.model
                print('F1 val improved --> storing best model')
                self.best_f1_val = self.model_args.f1_score[1]
                self.model_args.best_t = self.model_args.sweep_results['best_threshold']
                self.best_epoch = epoch
                self.best_model.save_weights(self.best_name)


    # Execute after last epoch    
    def on_train_end(self, logs={}):
        print('\nBest F1 val at epoch ', self.best_epoch+1,'.\n', sep='')
        return

##########################

class Experiment:

    def __init__(self, X, y, model_class):
        self.X = X
        self.y = y
        self.model_class = model_class
        
        # Get sparse y
        self.y_sparse = csr_matrix(self.y)

        # Make predictions
        self.y_pred = model_class.predict(self.X)


    def metrics(self, threshold = 0.5, k=None, avg='micro', verbose=1):

        def print_res(prec, rec, f1, avg):
            print(f'-- {avg} metrics --')
            print('F1\t\t\tPrecision\t\tRecall')
            print(f'{f1}\t{prec}\t{rec}')
        
        # From the chosen subsets:
        y_pred_bin = np.zeros(self.y_pred.shape)

        # If approach is fixed-k output
        if k:
            prob_rank = np.argsort(np.array(self.y_pred))

            for sample in range(len(y_pred_bin)):
                y_pred_bin[sample][prob_rank[sample][-k:]] = 1

        # Else if approach is threshold-based
        else:
            y_pred_bin = self.y_pred > threshold

        # Compute metrics
        self.precision, self.recall, self.f1_score, _ = precision_recall_fscore_support(self.y_sparse, csr_matrix(y_pred_bin), average=avg,zero_division=0)

        # Print metrics
        if verbose: print_res(self.precision, self.recall, self.f1_score, avg)


    def sweep_thresholds(self, thresholds = np.linspace(0.01,0.5,50), avg='micro', verbose=1):
    
        sweep_f1_val = []
        sweep_prec_val = []
        sweep_rec_val = []
        sweep_avg_val_pred = []

        # Sweep through thresholds
        for thresh in thresholds:

            self.metrics(threshold=thresh, avg=avg, verbose=0)

            sweep_f1_val.append(self.f1_score)
            sweep_prec_val.append(self.precision)
            sweep_rec_val.append(self.recall)
            sweep_avg_val_pred.append(np.mean(sum((self.y_pred>thresh).T)))

        best_t = thresholds[np.argmax(sweep_f1_val)]


        self.sweep_results = {'thresholds': thresholds,
                                'f1_val': sweep_f1_val,
                                'prec_val': sweep_prec_val,
                                'rec_val': sweep_rec_val,
                                'avg_val_pred': sweep_avg_val_pred,
                                'best_f1_val': self.f1_score,
                                'best_threshold': best_t,
                                }

        if verbose: 
            print(f'''
            Best Threshold: {best_t}
            ''')


    def sweep_k(self):
        pass





