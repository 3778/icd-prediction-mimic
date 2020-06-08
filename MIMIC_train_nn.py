## Train Neural Network models

import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
import pickle

from constants import SAVE_DIR, W2V_SIZE, MAX_LENGTH
import utils
import model_functions as fun


def main(args):
    ## 1. Load preprocessed data

    # Embedding layer
    wv_embedding_matrix, _ = utils.load_w2v_emb(w2v_vec_size=W2V_SIZE)

    # Correspondent preprocessed inputs and targets (x,y)
    # Note that x = [x_train, x_val, x_test] and y = [y_train, y_val, y_test]
    # This format fits into fun.model_args() class
    x, y, _ = utils.load_w2v_proc_inputs(max_words=MAX_LENGTH)


    ## 2. Instantiate model_args class

    model_args = fun.model_args(x,y)


    ## 3. Make model

    tf.keras.backend.clear_session()

    # Call model and compile
    model_args.model = utils.get_model(model_args, embedding_matrix=wv_embedding_matrix, args=args)

    if args.verbose: model_args.model.summary()
    
    # Instantiate callbacks
    # tensorboard_callback = TensorBoard(log_dir = SAVE_DIR + 'logs/fit/' + args.MODEL_NAME)
    f1_callback = fun.f1_callback_save_weights(model_args, best_name= SAVE_DIR + args.MODEL_NAME + '.h5')

    # callbacks = [tensorboard_callback, f1_callback]
    callbacks = [f1_callback]

    if args.MODEL_NAME == 'cnn_att':
        def scheduler(epoch):
            if epoch < 2:
                return 0.001    
            else:
                return 0.0001
        
        callbacks.append(LearningRateScheduler(scheduler, verbose=0))


    # Fit
    history = model_args.model.fit(model_args.x[0], model_args.y[0], validation_data=(model_args.x[1],model_args.y[1]),
                                epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=args.verbose)

    # Restore weights from the best epoch based on F1 val with optimized threshold
    best_model = utils.get_model(model_args, embedding_matrix=wv_embedding_matrix, args=args)
    best_model.load_weights(SAVE_DIR + args.MODEL_NAME + '.h5')

    # Predict
    model_args.predict(custom_model = best_model)

    # Compute metrics @ best threshold
    print(f'''
    --------------------
    Metrics @ {model_args.best_t:.2f} for best epoch:
    ''')
    model_args.metrics(threshold = model_args.best_t)

    # Save args to correctly load weights (this will go when I manage to correcly save models)
    with open(SAVE_DIR + 'args.pkl', 'wb') as file:
        pickle.dump(args, file)



## Parser
parser = argparse.ArgumentParser(description='Train model for MIMIC-III dataset and compute metrics.')
parser.add_argument('-model', type=str, dest='MODEL_NAME', choices=['cnn', 'gru','cnn_att'], default='cnn', help='Model for training.')
parser.add_argument('-epochs', type=int, dest='epochs', default=10, help='Number of epochs.')
parser.add_argument('-batch_size', type=int, dest='batch_size', default=16, help='Batch Size.')
parser.add_argument('-units', type=int, dest='units', default=500, help='Number of Units/Filters for training neural networks.')
parser.add_argument('-kernel_size', type=int, dest='kernel_size', default=10, help='Kernel size for CNNs.')
parser.add_argument('-lr', type=float, dest='lr', default=0, help='Learning rate for CNN and GRU. 0 for optimized values.')
parser.add_argument('-activation', type=str, dest='activation', default='tanh', help='Activation for CNN layers. CuDNNGRU must have tanh activation.')
parser.add_argument('--verbose', type=int, dest='verbose', default=2, help='Verbose when training.')

args = parser.parse_args()

# Start 
main(args)


