
# Predicting ICD Codes from Clinical Notes

This repository contains code for training and evaluating several neural network models for predicting ICD codes from discharge summaries on the publicly acessible [MIMIC-III](https://mimic.physionet.org/gettingstarted/overview/) dataset (v. 1.4). The models are described in the paper [Predicting Multiple ICD-10 Codes from Brazilian-Portuguese Clinical Notes](http://arxiv.org/abs/2008.01515), which uses the results on MIMIC-III as a benchmark. The implemented models are:

- Logistic Regression
- Convolutional Neural Network
- Recurrent Neural Network with Gated Recurrent Units
- Convolutional Neural Network with Attention (based on [CAML](https://github.com/jamesmullenbach/caml-mimic))


## Dependencies

This project depends on:

- python==3.6.9
- numpy==1.19.0
- scikit-learn==0.23.1
- pandas==0.25.3
- nltk==3.4.4
- scipy==1.4.1
- gensim==3.8.3
- tensorflow==2.1.0 (preferably tensorflow-gpu)


## General pipeline:


## 1. In `data/`, place the files below:
- NOTEEVENTS.csv.gz (from MIMIC-III)
- DIAGNOSES_ICD.csv.gz (from MIMIC-III)
- {train,dev,test}_full_hadm_ids.csv (all 3 from [CAML](https://github.com/jamesmullenbach/caml-mimic))

	
## 2. Run `MIMIC_preprocessing.py` to select discharge summaries and merge MIMIC-III tables.

MIMIC-III tables `NOTEEVENTS`, `DIAGNOSES_ICD` are loaded and merged through admission IDs. From `NOTEEVENTS`, only a single discharge summary is selected per admission ID.

Outputs `data/mimic3_data.pkl`, a DataFrame containing 4 columns:

- **HADM_ID**: the admission IDs of each patient stay. 
- **SUBJECT_ID**: the patient IDs. A patient may have multiple admissions, hence a `SUBJECT_ID` may be linked to multiple `HADM_IDs`.
- **TEXT**: the discharge summaries, one for each `HADM_ID`.
- **ICD9_CODE**: a list of ICD codes assigned to each stay (i.e. to each `HADM_ID`).

## 3. Run `MIMIC_train_w2v.py` to train Word2Vec word embeddings for the neural network models.

This script generates training instances by filtering `data/mimic3_data.pkl` with `data/train_full_hadmids.csv` to train *gensim.models.Word2Vec* word embeddings.

Outputs:
- **MIMIC_emb_train_vecW2V_SIZE.pkl**: an embedding matrix with shape *(vocab_length, embedding_dimension)*, in which every row contains the embedding vector of a word.
- **MIMIC_dict_train_vecW2V_SIZE.pkl**: a dictionary linking words to the respective row indexes in the embedding matrix.
- **w2v_model.model**: the trained Word2Vec instance.

## 4. Now, any model can be trained and evaluated:

### 4.1. Run `MIMIC_train_baselines.py`, for LR and Constant models.

- For Constant:
Computes the `k` most ocurring ICDs in the training set and predicts them for all test samples. Nothing is stored here.

- For LR:
Computes *TF-IDF* features in the training set. Then, fits the LR model to the training set.
After training, the weights of the epoch with best micro F1 in the validation set are restored and threshold-optimized metrics are computed for all subsets. The fitted model is stored using Tensorflow SavedModel format.


### 4.2. Run `MIMIC_train_nn.py`, for CNN, GRU and CNN-Att.

This script loads the data splits and Word2Vec embeddings, then fits the desired model for the training set.
After training, the weights of the epoch with best micro F1 in the validation set are restored and threshold-optimized metrics are computed for all subsets.

The fitted model is stored using Tensorflow SavedModel format.


## 5. In `notebooks/`, you will find:
- **MIMIC_overview.ipynb**, where some data analyses from MIMIC-III discharge summaries are provided.
- **MIMIC_analyze_predictions.ipynb**, where some additional analyses from the predictions of a trained model with W2V embeddings can be computed. The shown outputs are from our CNN-Att model. Edit the first cell with the desired model name and run all cells. 



