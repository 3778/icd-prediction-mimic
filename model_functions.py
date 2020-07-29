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
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model

import utils

class Experiments:

    def __init__(self, y_true, y_pred):
        """"
        Class for metrics computation.

        Inputs:
        y_true = [y_train, y_val, y_test]
        y_pred = [y_pred_train, y_pred_val, y_pred_test]
        
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_sparse = None

    def metrics(self, subsets = [1,1,1], avg='micro', threshold = 0.5, k=None, verbose=1):
        """ 
        Compute metrics.
        
        Parameters
        -----
        subsets: Choose which subsets to evaluate metrics [train, validation, test].\n        
        avg: average method to compute metrics in a multi-label problem.\n        
        threshold: if k=None, sets the threshold for binarizing model outputs.\n        
        k: if not None, sets the fixed length k of the outputs. The k highest scores from the model will be the ICD outputs.

        Outputs
        -----
        Outputs are stored in:

        self.f1_score[train, val, test]
        self.precision[train, val, test]
        self.recall[train, val, test]

        """
        self.f1_score = np.zeros(3)
        self.precision = np.zeros(3)
        self.recall = np.zeros(3)

        # Get sparse representation (faster)
        if self.y_sparse == None:
            self.y_sparse = [csr_matrix(self.y_true[i]) for i in range(3)]

        def print_res(prec, rec, f1, subset, avg):
            print('--%s %s-- metrics:' %(subset, avg))
            print('F1\t\t\tPrecision\t\tRecall')
            print(f1,'\t', prec, '\t',rec, sep='')
        
        subset = ['Train','Val','Test']

        y_pred_bin = [None]*3
        for i in range(3):

            # From the chosen subsets:
            if subsets[i]:
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


    def sweep_thresholds(self, subset=[0,1,0], thresholds = np.linspace(0.01,0.5,50), avg='micro', verbose=1):

        sweep_f1 = []
        sweep_prec = []
        sweep_rec = []
        sweep_avg_pred = []
        
        subset_idx = subset.index(1)

        # Sweep through thresholds
        for thresh in thresholds:
            
            self.metrics(subsets=subset, threshold=thresh, avg=avg, verbose=0)

            sweep_f1.append(self.f1_score[subset_idx])
            sweep_prec.append(self.precision[subset_idx])
            sweep_rec.append(self.recall[subset_idx])
            sweep_avg_pred.append(np.mean(sum((self.y_pred[subset_idx]>thresh).T)))

        best_t = thresholds[np.argmax(sweep_f1)]


        self.sweep_results = {'thresholds': thresholds,
                                'f1': sweep_f1,
                                'prec': sweep_prec,
                                'rec': sweep_rec,
                                'avg_pred': sweep_avg_pred,
                                'best_threshold': best_t,
                                }

        if verbose:
            print(f'''Best Threshold: {best_t:.2f}''')


    def sweep_k(self, subset=[0,1,0], ks = np.linspace(1,20,20, dtype='int32'), avg='micro', verbose=1):

        sweep_f1_val = []
        sweep_prec_val = []
        sweep_rec_val = []

        subset_idx = subset.index(1)

        # Sweep through k lengths
        for k in ks:

            self.metrics(subsets=subset, k=k, avg=avg, verbose=0)
            sweep_f1_val.append(self.f1_score[subset_idx])
            sweep_prec_val.append(self.precision[subset_idx])
            sweep_rec_val.append(self.recall[subset_idx])


        best_k = ks[np.argmax(sweep_f1_val)]

        if verbose:
            print(f'''Best k: {best_k}''')

        self.sweepk_results = {'ks': ks,
                                    'f1_val': sweep_f1_val,
                                    'prec_val': sweep_prec_val,
                                    'rec_val': sweep_rec_val,
                                    'best_k': best_k
                                    }










class f1_callback_save(Callback):
    """
    Callback for use in tf.keras. 
    Computes f1 @best_threshold in validation set and prints it.
    Optionally, it can save the best epoch model based on f1_val.
    If a tensorboard callback is specified, f1_val is stored in its logs.
    """

    def __init__(self, model, validation_data, tb_callback=None, store_best=True, best_name='best_model', avg='micro'):
        self.model = model
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.tb_callback = tb_callback
        self.avg = avg
        self.best_f1_val = 0
        self.store_best = store_best
        self.best_name = best_name

    # Execute after each epoch    
    def on_epoch_end(self, epoch, logs={}):

        # Predict and compute f1 val
        self.y_pred_val = self.model.predict(self.x_val)

        self.exp = Experiments([None,self.y_val, None], [None, self.y_pred_val, None])

        # sweep thresh
        self.exp.sweep_thresholds(subset=[0,1,0])

        # get metrics of best thresh
        self.exp.metrics(subsets=[0,1,0],threshold = self.exp.sweep_results['best_threshold'] )

        print('')
        
        # Set items to store in Tensorboard
        items_to_write={
            "f1_val": self.exp.f1_score[1]
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
            if self.exp.f1_score[1] > self.best_f1_val:
                self.best_model = self.model
                print('F1 val improved --> storing best model')
                self.best_f1_val = self.exp.f1_score[1]
                self.best_epoch = epoch
                self.best_model.save(self.best_name)

        

    # Execute after last epoch    
    def on_train_end(self, logs={}):
        print('\nBest F1 val at epoch ', self.best_epoch+1,'.\n', sep='')
        return