## Train Neural Network models

import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import pickle

from constants import SAVE_DIR, W2V_SIZE, MAX_LENGTH

import datasets
import feature_extraction as fx
import model_functions as fun
import utils

def main(args):

    # Clear session
    tf.keras.backend.clear_session()

    # Load data and embeddings
    mimic = datasets.MIMIC_Dataset()
    mimic.load_preprocessed()
    mimic.split()

    embedding = fx.W2V(args)
    embedding.load_embedding(dataset_name=mimic.name)

    embedding.transform(dataset=mimic)

    # Call model class
    model = utils.get_model(args)
    

    # Instantiate callback
    f1_callback = fun.f1_callback_save(model, validation_data=(embedding.x_val, mimic.y_val),
                                       best_name= SAVE_DIR + args.MODEL_NAME)

    callbacks = [f1_callback]

    if args.schedule_lr: 
        def scheduler(epoch):
            if epoch < args.epoch_drop:
                return args.initial_lr   
            else:
                return args.final_lr
        
        callbacks.append(LearningRateScheduler(scheduler, verbose=0))

    # Fit
    model.fit(embedding.x_train, mimic.y_train, embedding.embedding_matrix, validation_data=(embedding.x_val, mimic.y_val), callbacks=callbacks)

    # Restore weights from the best epoch based on F1 val with optimized threshold
    ### obs: not keeping last epoch, only best one. Maybe also save last epoch for further training? (where to add this?)
    model.load(path=SAVE_DIR + args.MODEL_NAME)

    # Predict
    y_pred_train = model.predict(embedding.x_train)
    y_pred_val = model.predict(embedding.x_val)
    y_pred_test = model.predict(embedding.x_test)

    exp = fun.Experiments(y_true = [mimic.y_train, mimic.y_val, mimic.y_test],
                          y_pred = [y_pred_train, y_pred_val, y_pred_test])

    # Compute best threshold
    exp.sweep_thresholds(subset=[0,1,0])

    print(f'''
    Metrics @ {exp.sweep_results['best_threshold']}''')
    # Compute metrics @ best threshold
    exp.metrics(threshold=exp.sweep_results['best_threshold'])


## Parser
parser = argparse.ArgumentParser(description='Train model for MIMIC-III dataset and compute metrics.')
parser.add_argument('-model', type=str, dest='MODEL_NAME', choices=['cnn', 'gru','cnn_att'], default='cnn', help='Model for training.')
parser.add_argument('-epochs', type=int, dest='epochs', default=10, help='Number of epochs.')
parser.add_argument('-batch_size', type=int, dest='batch_size', default=16, help='Batch Size.')
parser.add_argument('-units', type=int, dest='units', default=500, help='Number of Units/Filters for training neural networks.')
parser.add_argument('-kernel_size', type=int, dest='kernel_size', default=10, help='Kernel size for CNNs.')
parser.add_argument('-lr', type=float, dest='lr', default=0, help='Learning rate for CNN and GRU. 0 for optimized values.')
parser.add_argument('-schedule_lr', type=float, dest='schedule_lr', default=0, help='Wether to use learning rate schedule with step decay. Set to 1 for CNN_att optimized model.')
parser.add_argument('-initial_lr', type=float, dest='initial_lr', default=0.001, help='Starting lr for schedule. Leave default for CNN_att optimized value.')
parser.add_argument('-final_lr', type=float, dest='final_lr', default=0.0001, help='Ending lr for schedule. Leave default for CNN_att optimized value.')
parser.add_argument('-epoch_drop', type=int, dest='epoch_drop', default=2, help='Epoch where lr schedule will shift initial_lr by final_lr. Leave default for CNN_att optimized value.')
parser.add_argument('-activation', type=str, dest='activation', default='tanh', help='Activation for CNN layers. CuDNNGRU must have tanh activation.')
parser.add_argument('--verbose', type=int, dest='verbose', default=2, help='Verbose when training.')

args = parser.parse_args()

# Start 
main(args)
