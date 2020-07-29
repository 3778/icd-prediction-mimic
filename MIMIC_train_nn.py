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

## Train Neural Network models

import argparse
import tensorflow as tf
import pickle

from constants import SAVE_DIR, W2V_SIZE, MAX_LENGTH

import datasets
import feature_extraction as fx
import model_functions as fun
import utils

def main(args):

    save_path = SAVE_DIR + args.MODEL_NAME

    # Clear session
    tf.keras.backend.clear_session()

    # Load data and embeddings
    mimic = datasets.MIMIC_Dataset()
    mimic.load_preprocessed()
    mimic.split()

    # Load trained embedding
    embedding = fx.W2V('MIMIC')

    # Transform using input
    embedding.transform(mimic)

    # Call model class
    model = utils.get_model(args)
    

    # Instantiate callback
    f1_callback = fun.f1_callback_save(model, validation_data=(embedding.x_val, mimic.y_val),
                                    best_name = save_path)

    callbacks = [f1_callback]

    # Learning rate single-step schedule
    if args.schedule_lr: 
        callbacks.append(utils.lr_schedule_callback(args))

    # Fit
    model.fit(embedding.x_train, mimic.y_train, embedding.embedding_matrix, validation_data=(embedding.x_val, mimic.y_val), callbacks=callbacks)

    # Save model state after last epoch
    if args.save_last_epoch:
        model.save_model(f'{save_path}ep{args.epochs}')

    # Restore weights from the best epoch based on F1 val with optimized threshold
    model = utils.get_model(args, load_path = save_path)

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


def arg_parser():
    
    parser = argparse.ArgumentParser(description='Train model for MIMIC-III dataset and compute metrics.')
    parser.add_argument('-model', type=str, dest='MODEL_NAME', choices=['cnn', 'gru','cnn_att'], default='cnn', help='Model for training.')
    parser.add_argument('-epochs', type=int, dest='epochs', default=10, help='Number of epochs.')
    parser.add_argument('-batch_size', type=int, dest='batch_size', default=16, help='Batch Size.')
    parser.add_argument('-units', type=int, dest='units', default=500, help='Number of Units/Filters for training neural networks.')
    parser.add_argument('-kernel_size', type=int, dest='kernel_size', default=10, help='Kernel size for CNNs.')
    parser.add_argument('-lr', type=float, dest='lr', default=0, help='Learning rate for CNN and GRU. 0 for optimized values.')
    parser.add_argument('-schedule_lr', type=bool, dest='schedule_lr', default=False, help='Wether to use learning rate schedule with step decay. Set to 1 for CNN_att optimized model.')
    parser.add_argument('--initial_lr', type=float, dest='initial_lr', default=0.001, help='Starting lr for schedule. Leave default for CNN_att optimized value.')
    parser.add_argument('--final_lr', type=float, dest='final_lr', default=0.0001, help='Ending lr for schedule. Leave default for CNN_att optimized value.')
    parser.add_argument('--epoch_drop', type=int, dest='epoch_drop', default=2, help='Epoch where lr schedule will shift from initial_lr to final_lr. Leave default for CNN_att optimized value.')
    parser.add_argument('-activation', type=str, dest='activation', default='tanh', help='Activation for CNN layers. CuDNNGRU must have tanh activation.')
    parser.add_argument('-save_last_epoch', type=bool, dest='save_last_epoch', default=False, help='Also save model state at last epoch (additionally to best epoch)')
    parser.add_argument('--verbose', type=int, dest='verbose', default=2, help='Verbose when training.')

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parser()

    main(args)