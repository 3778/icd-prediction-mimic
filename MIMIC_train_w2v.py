### Train Word2Vec word vectors for the MIMIC-III dataset.

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
    w2v.fit(mimic.x_train)

    # Save embedding matrix
    w2v.save_embedding(dataset_name=mimic.name)

    print(f'''
    Word2Vec embeddings saved!
    ''')


## Parser
parser = argparse.ArgumentParser(description='Train Word2Vec word embeddings')
parser.add_argument('-workers', type=int, dest='workers', default=8, help='Number of CPU threads for W2V training.')
parser.add_argument('--reset_stopwords', type=bool, dest='reset_stopwords', default=0, help='True to set stopwords vectors to null. Default False.')
parser.add_argument('--train_method', type=bool, dest='sg', default=1, help='W2V train method. 0 for CBoW, 1 for Skipgram.')

args = parser.parse_args()

# Start 
main(args)