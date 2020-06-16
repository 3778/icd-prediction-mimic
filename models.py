import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, GlobalAveragePooling1D, BatchNormalization #, CuDNNGRU
from tensorflow.keras.layers import Layer, Attention
from tensorflow.keras.optimizers import Adam

######## LR MODEL ######## 

def lr_model(model_args, args):

    if not args.lr:
        args.lr = 0.1
    
    input_shape = model_args.x[0].shape[1]
    output_shape = model_args.y[0].shape[1]
     
    # Create LR
    inputs = Input(shape=(input_shape,))

    outputs = Dense(output_shape, activation='sigmoid')(inputs)

    model_args.model = Model(inputs, outputs)
    model_args.model.compile(loss='binary_crossentropy',optimizer=Adam(args.lr),metrics=['acc'])
        
    return model_args.model



######## CNN MODEL ######## 

def cnn_model(model_args, embedding_matrix, args):
    
    output_shape = model_args.y[0].shape[1]
    
    if not args.lr:
        args.lr = 0.001

    # Define model
    sequence_input = Input(shape=(model_args.x[0].shape[1],), dtype='int32')
    
    embedding_layer = Embedding(input_dim = embedding_matrix.shape[0], 
                                output_dim = embedding_matrix.shape[1], 
                                weights = [embedding_matrix], 
                                input_length = model_args.x[0].shape[1],
                                trainable = True) (sequence_input)
    
    H = Conv1D(args.units, args.kernel_size, activation=args.activation, padding='same')(embedding_layer)
    H = BatchNormalization() (H)
    
    x = GlobalAveragePooling1D()(H)
    preds = Dense(output_shape, activation='sigmoid')(x)
    
    model_args.model = Model(sequence_input, preds)
    model_args.model.compile(loss='binary_crossentropy',optimizer=Adam(args.lr),metrics=['acc'])
    
    return model_args.model


######## GRU MODEL ######## 

# def gru_model(model_args, embedding_matrix, args):

#     # Parameters
#     output_shape = model_args.y[0].shape[1]

#     if not args.lr:
#         args.lr = 8e-4

#     # Build model
#     sequence_input = Input(shape=(model_args.x[0].shape[1],), dtype='int32')

#     embedding_layer = Embedding(input_dim = embedding_matrix.shape[0], 
#                                 output_dim = embedding_matrix.shape[1], 
#                                 weights = [embedding_matrix], 
#                                 input_length = model_args.x[0].shape[1],
#                                 trainable = True) (sequence_input)

#     x = CuDNNGRU(args.units, return_sequences=True) (embedding_layer)
#     x = BatchNormalization() (x)
#     x = GlobalAveragePooling1D() (x)

#     outputs = Dense(output_shape, activation='sigmoid') (x)

#     model_args.model = Model(sequence_input, outputs)
#     model_args.model.compile(loss='binary_crossentropy',optimizer=Adam(args.lr),metrics=['acc'])

#     return model_args.model



######## CNN-Att MODEL ######## 

# Custom layer to generate per-label weights
class TrainableMatrix(Layer):
    def __init__(self, n_rows, n_cols, **kwargs):
        super().__init__(**kwargs)
        self.n_rows = n_rows
        self.n_cols = n_cols
    def build(self, input_shape):
        self.U = self.add_weight(name='trainmat', shape=(self.n_rows, self.n_cols), initializer='glorot_uniform', trainable=True)
        super(TrainableMatrix, self).build(input_shape)
    def call(self, inputs):
        return self.U
    
    def get_config(self):
        config = super(TrainableMatrix, self).get_config()
        config.update({
            'n_rows': self.n_rows,
            'n_cols': self.n_cols
        })
        return config


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
        super(Hadamard, self).build(input_shape)
    def call(self, x):
        return tf.keras.activations.sigmoid(tf.reduce_sum(x*self.kernel + self.bias, axis=-1))
    def compute_output_shape(self, input_shape):
        return input_shape

# Main model
def cnn_att_model(model_args, embedding_matrix, args):

    output_shape = model_args.y[0].shape[1]
    optimizer = Adam() # lr not specified as it is scheduled

    # Define model
    sequence_input = Input(shape=(model_args.x[0].shape[1],), dtype='int32')
    
    embedding_layer = Embedding(input_dim = embedding_matrix.shape[0], 
                                output_dim = embedding_matrix.shape[1], 
                                weights = [embedding_matrix], 
                                input_length = model_args.x[0].shape[1],
                                trainable = True) (sequence_input)
    
    H = Conv1D(args.units, args.kernel_size, activation=args.activation, padding='same')(embedding_layer)
    H = BatchNormalization() (H)

    U = TrainableMatrix(output_shape,args.units)([])
    
    att = Attention(use_scale=True) ([U, H])

    preds = Hadamard() (att)

    model_args.model = Model(sequence_input, preds)
    model_args.model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    
    return model_args.model

##########################

class CTE_Model:
    def __init__(self):
        pass

class LR_Model:
    
    def __init__(self,args):
        self.args = args

    def lr_model(self, input_shape, output_shape):

        if not self.args.lr: ## how to instantiate default values
            self.args.lr = 0.1
            
        # Create LR
        inputs = Input(shape=(input_shape,))

        outputs = Dense(output_shape, activation='sigmoid')(inputs)

        model = Model(inputs, outputs)
        model.compile(loss='binary_crossentropy',optimizer=Adam(self.args.lr))
            
        return model
    

    def fit(self, X, y, validation_data=None, callbacks=None):

        self.model = self.lr_model(len(X[0]), len(y[0]))

        if self.args.verbose: self.model.summary()

        self.model.fit(X, y, validation_data=validation_data, 
                       epochs=self.args.epochs, batch_size=self.args.batch_size, 
                       callbacks=callbacks, verbose=self.args.verbose)

    def load(self, path):
        self.model = load_model(path)

    def predict(self, X):
        return self.model.predict(X)


class CNN_Model:

    def __init__(self,args):
        self.args = args

    def cnn_model(self, input_shape, output_shape, embedding_matrix):
            
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
        
        x = GlobalAveragePooling1D()(H)
        preds = Dense(output_shape, activation='sigmoid')(x)
        
        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',optimizer=Adam(self.args.lr),metrics=['acc'])
        
        return model

    def fit(self, X, y, embedding_matrix, validation_data=None, callbacks=None):
    
        # self.model = self.cnn_model(X.shape[1], y.shape[1], embedding_matrix)
        self.model = self.cnn_model(len(X[0]), len(y[0]), embedding_matrix)

        if self.args.verbose: self.model.summary()

        self.model.fit(X, y, validation_data=validation_data, 
                       epochs=self.args.epochs, batch_size=self.args.batch_size, 
                       callbacks=callbacks, verbose=self.args.verbose)

    def load(self, path):
        self.model = load_model(path)

    def predict(self, X):
        return self.model.predict(X)


class GRU_Model:
    pass

class CNNAtt_Model:
    pass



