
# Predicting Multiple ICD-10 Codes from Brazilian-Portuguese Clinical Notes


This repository contains code for training and evaluating all models described in the paper [link2paper](url), for the publicly acessible [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/) dataset (v. 1.4).


## Dependencies

???

To install dependencies, run ?


## General pipeline:


### 1. In `data/`, place the files below:
- NOTEEVENTS.csv.gz (from MIMIC-III)
- DIAGNOSES_ICD.csv.gz (from MIMIC-III)
- {train,dev,test}_full_hadm_ids.csv (all 3 from [CAML](https://github.com/jamesmullenbach/caml-mimic))

	
### 2. Run MIMIC_preprocessing.py to select discharge summaries and merge MIMIC-III tables.

MIMIC-III tables `NOTEEVENTS`, `DIAGNOSES_ICD` are loaded and joined through admission IDs. From `NOTEEVENTS`, only a single discharge summary is selected per admission ID.

Outputs `data/mimic3_data.pkl`, a DataFrame containing 4 columns:

- **HADM_ID**: the admission IDs of each patient stay. 
- **SUBJECT_ID**: the patient IDs. A patient may have multiple admissions, hence a `SUBJECT_ID` may be linked to multiple `HADM_IDs`.
- **TEXT**: the discharge summaries, one for each HADM_ID.
- **ICD9_CODE**: a list of ICD codes assigned to each stay (i.e. to each `HADM_ID`).

### 3. Run MIMIC_train_w2v.py to train Word2Vec word embeddings for the neural network models.

This script generates training instances by filtering `data/mimic3_data.pkl` with `data/train_full_hadmids.csv` to train *gensim.models.Word2Vec* word embeddings.

Outputs:
- **MIMIC_emb_train_vecW2V_SIZE.pkl**: an embedding matrix with shape *(vocab_length, embedding_dimension)*, in which every row contains the embedding vector of a word.
- **MIMIC_dict_train_vecW2V_SIZE.pkl**: a dictionary linking words to the respective row indexes in the embedding matrix.
- **w2v_model.model**: the trained Word2Vec instance.

### 4. Now, any model can be trained and evaluated:

#### 4.1. Run MIMIC_train_baselines.py, for LR and Constant models.

- For Constant:
Computes the `k` most ocurring ICDs in the training set and predicts them for all test samples. Nothing is stored here.

- For LR:
Computes *TF-IDF* features in the training set. Then, fits the LR model to the training set.
After training, the weights of the epoch with best micro F1 in the validation set are restored and threshold-optimized metrics are computed for all subsets.

Here, the fitted model is stored using Tensorflow SavedModel format.


#### 4.2. Run MIMIC_train_nn.py, for CNN, GRU and CNN-Att.

This script loads the data splits and Word2Vec embeddings, then fits the desired model for the training set.
After training, the weights of the epoch with best micro F1 in the validation set are restored and threshold-optimized metrics are computed for all subsets.

The fitted model is stored using Tensorflow SavedModel format.


### 5. In /notebooks/, you will find:
- *MIMIC_overview.ipynb*, where some data analysis from MIMIC-III discharge summaries are provided.
- *MIMIC_analyze_predictions.ipynb*, where some additional analyses from the predictions of a model can be seen.


## Analyze trained models:

Our trained models are provided at [physionet_model_link](url). To get predictions and metrics for them:

- Place the desired models in *models/*.
- Run steps 1 to 4. 
- In notebook *notebooks/MIMIC_analyze_predictions.ipynb*, select a model and run all cells.

