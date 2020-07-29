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

## Train Logistic Regression and Constant models

import argparse
import tensorflow as tf

from constants import DATA_DIR, SAVE_DIR
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

    # Get model class
    model = utils.get_model(args)

    if args.MODEL_NAME == 'lr':
        tfidf = fx.TFIDF(args)
        tfidf.fit(mimic)
        tfidf.transform(mimic)

        # Instantiate callback
        f1_callback = fun.f1_callback_save(model, validation_data=(tfidf.x_val, mimic.y_val),
                                        best_name= save_path)

        callbacks = [f1_callback]    

        
        # Fit
        model.fit(tfidf.x_train, mimic.y_train, validation_data=(tfidf.x_val, mimic.y_val), callbacks=callbacks)


        # Save model state after last epoch
        if args.save_last_epoch:
            model.save_model(f'{save_path}ep{args.epochs}')

        # Restore weights from the best epoch based on F1 val with optimized threshold
        model = utils.get_model(args, load_path = save_path)

        # Predict
        y_pred_train = model.predict(tfidf.x_train)
        y_pred_val = model.predict(tfidf.x_val)
        y_pred_test = model.predict(tfidf.x_test)


        exp = fun.Experiments(y_true = [mimic.y_train, mimic.y_val, mimic.y_test],
                               y_pred = [y_pred_train, y_pred_val, y_pred_test])

        # Compute best threshold
        exp.sweep_thresholds(subset=[0,1,0])

        print(f'''
        Metrics @ {exp.sweep_results['best_threshold']}''')
        # Compute metrics @ best threshold
        exp.metrics(threshold=exp.sweep_results['best_threshold'])  


    elif args.MODEL_NAME == 'cte':

        # model.fit(mimic.y_train, most_occ_train=mimic.all_icds_train)
        model.fit(most_occ_train=mimic.all_icds_train) 

        # Predict
        y_pred_train = model.predict(mimic.x_train, mlb=mimic.mlb)
        y_pred_val = model.predict(mimic.x_val, mlb=mimic.mlb)
        y_pred_test = model.predict(mimic.x_test, mlb=mimic.mlb)

        exp = fun.Experiments(y_true = [mimic.y_train, mimic.y_val, mimic.y_test],
                               y_pred = [y_pred_train, y_pred_val, y_pred_test])

        print(f"""
        Metrics @ {args.k}""")
        # Compute metrics @ k
        exp.metrics(k=args.k)   



def arg_parser():

    parser = argparse.ArgumentParser(description='Train model for MIMIC-III dataset and compute metrics.')
    parser.add_argument('-model', type=str, dest='MODEL_NAME', choices=['lr', 'cte'], default = 'lr',help='Model for training.')
    parser.add_argument('-epochs', type=int, dest='epochs', default=10, help='Number of epochs.')
    parser.add_argument('-tfidf_maxfeatures', type=int, dest='max_features', default=20000, help='Max features for TF-IDF.')
    parser.add_argument('-batch_size', type=int, dest='batch_size', default=32, help='Batch Size.')
    parser.add_argument('-lr', type=float, dest='lr', default=0, help='Learning Rate. 0 for article optimized value.')
    parser.add_argument('-k', type=int, dest='k', default=15, help='Fixed k-size of predictions for Constant Model.')
    parser.add_argument('-save_last_epoch', type=bool, dest='save_last_epoch', default=False, help='Also save model state at last epoch (additionally to best epoch)')
    parser.add_argument('--verbose', type=int, dest='verbose', default=2, help='Verbose when training.')

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parser()

    main(args)