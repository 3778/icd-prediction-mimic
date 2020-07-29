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

## Train Word2Vec word vectors for the MIMIC-III dataset.

import argparse
import datasets
import feature_extraction as fx

def main(args):

    # Load dataset
    mimic = datasets.MIMIC_Dataset()
    mimic.load_preprocessed()
    mimic.split()
    
    # Instantiate embedding
    w2v = fx.W2V(args)

    # Train
    w2v.fit(mimic)

    # Save embedding matrix
    w2v.save_embedding(dataset_name=mimic.name)

    print(f'''
        Word2Vec embeddings saved!
    ''')


def arg_parser():

    parser = argparse.ArgumentParser(description='Train Word2Vec word embeddings')
    parser.add_argument('-workers', type=int, dest='workers', default=8, help='Number of CPU threads for W2V training.')
    parser.add_argument('--reset_stopwords', type=bool, dest='reset_stopwords', default=0, help='True to set stopwords vectors to null. Default False.')
    parser.add_argument('--train_method', type=bool, dest='sg', default=1, help='W2V train method. 0 for CBoW, 1 for Skipgram.')

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parser()

    main(args)