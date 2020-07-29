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

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, GlobalAveragePooling1D, BatchNormalization, GRU
from tensorflow.keras.layers import Layer, Attention
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
import utils

# Try to import CuDNNGRU, otherwise use regular GRU
try:
   from tensorflow.keras.layers import CuDNNGRU as GRU
except ImportError:
   no_requests = True



class CTE_Model:

    def __init__(self, args, load_path=None):
        self.args = args

    def cte_model(self):
        pass
    
    def fit(self, most_occ_train=None): # CTE_Model does not use X data 
        
        # Select most occuring ICDs in train set
        self.most_occ_train = most_occ_train[:self.args.k]

    def predict(self, X, mlb):

        y_pred = np.repeat([self.most_occ_train[:self.args.k]],repeats=len(X),axis=0)
        y_pred = mlb.transform(y_pred)
        
        return y_pred
        


class LR_Model:
    
    def __init__(self, args=None, load_path=None):

        # You could load from path but also get args, in order to continue training. 
        self.args = args 
        self.model = None

        if load_path:
            self.load_path = load_path
            self.load_from_path()
            

    def load_from_path(self):
        self.model = load_model(self.load_path)

    def lr_model(self, input_shape, output_shape):

        if not self.args.lr: ### check how to instantiate default values
            self.args.lr = 0.1
            
        # Create LR
        inputs = Input(shape=(input_shape,))

        outputs = Dense(output_shape, activation='sigmoid')(inputs)

        model = Model(inputs, outputs)
        model.compile(loss='binary_crossentropy',optimizer=Adam(self.args.lr))
            
        return model
    
    def fit(self, X, y, validation_data=None, callbacks=None):

        if not self.model:
            self.model = self.lr_model(X[0].shape[0], y[0].shape[0])

        if self.args.verbose: self.model.summary()

        self.model.fit(X, y, validation_data=validation_data, 
                       epochs=self.args.epochs, batch_size=self.args.batch_size, 
                       callbacks=callbacks, verbose=self.args.verbose)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        # No need to save model if f1_callback is used, as it already saved model at best epoch
        self.model.save(path)



class CNN_Model:

    def __init__(self, args=None, load_path=None):
        self.args = args
        self.model = None

        if load_path:
            self.load_path = load_path
            self.load_from_path()

    def load_from_path(self):
        self.model = load_model(self.load_path)

    def cnn_model(self, input_shape, output_shape, embedding_matrix):
            
        if not self.args.lr:
            self.args.lr = 0.001

        # Define model
        sequence_input = Input(shape=(input_shape,),) #dtype='int32'
        
        embedding_layer = Embedding(input_dim = embedding_matrix.shape[0], 
                                    output_dim = embedding_matrix.shape[1], 
                                    weights = [embedding_matrix], 
                                    input_length = input_shape,
                                    trainable = True) (sequence_input)
        
        H = Conv1D(self.args.units, self.args.kernel_size, activation=self.args.activation, padding='same')(embedding_layer)
        H = BatchNormalization() (H)
        
        x = GlobalAveragePooling1D()(H)
        preds = Dense(output_shape, activation='sigmoid')(x)
        
        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',optimizer=Adam(self.args.lr))
        
        return model

    def fit(self, X, y, embedding_matrix, validation_data=None, callbacks=None):
    
        if not self.model:
            self.model = self.cnn_model(X[0].shape[0], y[0].shape[0], embedding_matrix)

        if self.args.verbose: self.model.summary()

        self.model.fit(X, y, validation_data=validation_data, 
                       epochs=self.args.epochs, batch_size=self.args.batch_size, 
                       callbacks=callbacks, verbose=self.args.verbose)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        # No need to save model if f1_callback is used, as it already saved model at best epoch
        self.model.save(path)



class GRU_Model:

    def __init__(self,args=None, load_path=None):
        self.args = args
        self.model = None

        if load_path:
            self.load_path = load_path
            self.load_from_path()

    def load_from_path(self):
        self.model = load_model(self.load_path)

    def gru_model(self, input_shape, output_shape, embedding_matrix):

        if not self.args.lr:
            self.args.lr = 8e-4

        # Build model
        sequence_input = Input(shape=(input_shape,), dtype='int32')

        embedding_layer = Embedding(input_dim = embedding_matrix.shape[0], 
                                    output_dim = embedding_matrix.shape[1], 
                                    weights = [embedding_matrix], 
                                    input_length = input_shape,
                                    trainable = True) (sequence_input)

        # Note that if cudnn is available CuDNNGRU will be used for faster training
        x = GRU(self.args.units, return_sequences=True) (embedding_layer)

        x = BatchNormalization() (x)
        x = GlobalAveragePooling1D() (x)

        outputs = Dense(output_shape, activation='sigmoid') (x)

        model = Model(sequence_input, outputs)
        model.compile(loss='binary_crossentropy',optimizer=Adam(self.args.lr))

        return model

    def fit(self, X, y, embedding_matrix, validation_data=None, callbacks=None):

        if not self.model:
            # self.model = self.gru_model(X.shape[1], y.shape[1], embedding_matrix)
            self.model = self.gru_model(X[0].shape[0], y[0].shape[0], embedding_matrix)

        if self.args.verbose: self.model.summary()

        self.model.fit(X, y, validation_data=validation_data, 
                        epochs=self.args.epochs, batch_size=self.args.batch_size, 
                        callbacks=callbacks, verbose=self.args.verbose)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        # No need to save model if f1_callback is used, as it already saved model at best epoch
        self.model.save(path)



class CNNAtt_Model:

    def __init__(self,args=None, load_path=None):
        self.args = args
        self.model = None

        if load_path:
            self.load_path = load_path
            self.load_from_path()

    def load_from_path(self):
        self.model = load_model(self.load_path)

    #Custom layer to generate per-label weights
    class TrainableMatrix(Layer):
        def __init__(self, n_rows, n_cols, **kwargs):
            super().__init__(**kwargs)
            self.n_rows = n_rows
            self.n_cols = n_cols
        def build(self, input_shape):
            self.U = self.add_weight(name='trainmat', shape=(self.n_rows, self.n_cols), initializer='glorot_uniform', trainable=True)
            # super(self.TrainableMatrix, self).build(input_shape)
            super().build(input_shape)
        def call(self, inputs):
            return self.U

    # Custom layer to apply a LR for each label and then concatenate predictions
    class Hadamard(Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def build(self, input_shape):
            self.kernel = self.add_weight(name='kernel',
                                        shape=(1,) + tuple([int(a) for a in input_shape[1:]]),
                                        initializer='glorot_uniform',
                                        trainable=True)
            self.bias = self.add_weight(name='bias',
                                        shape=(1,) + tuple([int(a) for a in input_shape[1:]]),
                                        initializer='zeros',
                                        trainable=True)
            # super(self.Hadamard, self).build(input_shape)
            super().build(input_shape)
        def call(self, x):
            return tf.keras.activations.sigmoid(tf.reduce_sum(x*self.kernel + self.bias, axis=-1))
        def compute_output_shape(self, input_shape):
            return input_shape

    # Main model
    def cnn_att_model(self, input_shape, output_shape, embedding_matrix):

        if not self.args.lr:
            self.args.lr = 0.001

        # Define model
        sequence_input = Input(shape=(input_shape,), dtype='int32')
        
        embedding_layer = Embedding(input_dim = embedding_matrix.shape[0], 
                                    output_dim = embedding_matrix.shape[1], 
                                    weights = [embedding_matrix], 
                                    input_length = input_shape,
                                    trainable = True) (sequence_input)
        
        H = Conv1D(self.args.units, self.args.kernel_size, activation=self.args.activation, padding='same')(embedding_layer)
        H = BatchNormalization() (H)

        U = self.TrainableMatrix(output_shape,self.args.units)([])
        
        att = Attention(use_scale=True) ([U, H])

        preds = self.Hadamard() (att)

        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',optimizer=Adam(self.args.lr))
        
        return model

    def fit(self, X, y, embedding_matrix, validation_data=None, callbacks=None):

        if not self.model:
            # self.model = self.cnn_att_model(X.shape[1], y.shape[1], embedding_matrix)
            self.model = self.cnn_att_model(X[0].shape[0], y[0].shape[0], embedding_matrix)

        if self.args.verbose: self.model.summary()

        self.model.fit(X, y, validation_data=validation_data, 
                        epochs=self.args.epochs, batch_size=self.args.batch_size, 
                        callbacks=callbacks, verbose=self.args.verbose)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        # No need to save model if f1_callback is used, as it already saved model at best epoch
        self.model.save(path)



