
# Predicting Multiple ICD-10 Codes from Brazilian-Portuguese Clinical Notes


This repository contains code for training and evaluating all models described in the paper [link2paper](url), for the publicly acessible MIMIC-III dataset (v. 1.4).


## Dependencies

???

To install dependencies, run ?


## General pipeline:


### 1. Edit constants.py to point to the correct directories.

### 2. In data/, place the files below:
	- NOTEEVENTS.csv.gz (from MIMIC-III)
	- DIAGNOSES_ICD.csv.gz (from MIMIC-III)
	- *_full_hadm_ids.csv (from CAML)

Get the hadm_ids.csv files from [CAML](https://github.com/jamesmullenbach/caml-mimic). You will also need access to the [MIMIC-III dataset](https://mimic.physionet.org/gettingstarted/access/).
	
### 3. Run MIMIC_preprocessing.py to select discharge summaries and merge MIMIC-III tables.

This script loads the MIMIC-III tables *NOTEEVENTS*, *DIAGNOSES_ICD* and links them through admission IDs. From *NOTEEVENTS*, only a single discharge summary is selected per admission ID.

This script generates *data/mimic3_data.pkl*, a DataFrame containing 4 columns:

- **HADM_ID**: the admission IDs of each patient stay. 
- **SUBJECT_ID**: the patient IDs. A patient may have multiple admissions, hence a *SUBJECT_ID* may be linked to multiple *HADM_IDs*.
- **TEXT**: the discharge summaries, one for each HADM_ID.
- **ICD9_CODE**: a list of ICD codes assigned to each stay (i.e. to each *HADM_ID*).

### 4. Run MIMIC_train_w2v.py to train Word2Vec word embeddings for the neural network models.

This script takes *data/mimic3_data.pkl* and gets the training split (selecting the *HADM_IDs* contained in *data/train_full_hadmids.csv*).
Then, it instantiates *gensim.models.Word2Vec* class and trains the embedding with that split.

The outputs from this script are:
- **MIMIC_emb_train_vecW2V_SIZE.pkl**: an embedding matrix with shape *(vocab_length, embedding_dimension)*, in which every row contains the embedding vector of a word.
- **MIMIC_dict_train_vecW2V_SIZE.pkl**: a dictionary linking words to the respective row indexes in the embedding matrix.
- **w2v_model.model**: the trained Word2Vec instance.

### 6. Now, any model can be trained and evaluated:

#### 6.1. Run MIMIC_train_baselines.py, for LR and Constant models.

- For Constant:
The script computes the `k` most ocurring ICDs in the training set and predicts them to all samples. Nothing is stored here.

- For LR:
This script computes TF-IDF features in the training set. Then, it fits an LR model to the training set.
After training, it restores the weights of the epoch with best micro F1 in the validation set.
Finally, it computes the threshold-optimized metrics in all subsets.

Here, the script saves the fitted model, using Tensorflow SavedModel format.


#### 6.2. Run MIMIC_train_nn.py, for CNN, GRU and CNN-Att.

This script takes the data splits and Word2Vec embeddings, then fits the desired model for the training set.
After training, it restores the weights of the epoch with best micro F1 in the validation set.
Finally, it computes the threshold-optimized metrics in all subsets.

The fitted model is stored using Tensorflow SavedModel format.


### 7. In /notebooks/, you will find:
- *MIMIC_overview.ipynb*, where some data analysis from MIMIC-III discharge summaries are provided.
- *MIMIC_analyze_predictions.ipynb*, where some additional analyses from the predictions of a model can be seen.


## Analyze trained models:

Our trained models are provided at [physionet_model_link](url). To get predictions and metrics for them:

- Place the desired models in *models/*.
- Run steps 1 to 4. 
- In notebook *notebooks/MIMIC_analyze_predictions.ipynb*, select a model and run all cells.

