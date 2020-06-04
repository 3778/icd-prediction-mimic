
# Predicting Multiple ICD-10 Codes from Brazilian-Portuguese Clinical Notes


This repository contains code for training and evaluating all models described in the paper [link2paper](url), for the publicly acessible MIMIC-III dataset (v. 1.4).


## Dependencies

???

To install dependencies, run ?


## General pipeline:


1. Edit constants.py to match the repo directory.

2. In /data/, place the files below:
	- NOTEEVENTS.csv.gz (from MIMIC-III)
	- DIAGNOSES_ICD.csv.gz (from MIMIC-III)
	- *_full_hadm_ids.csv (from [CAML](https://github.com/jamesmullenbach/caml-mimic))
	
3. Run MIMIC_preprocessing.py to select discharge summaries and merge MIMIC-III tables.

4. Run MIMIC_w2v_train.py to train Word2Vec word embeddings for the neural network models.

5. Run MIMIC_process_inputs.py in order to process the inputs to match the embedding matrix and multi-hot encode the targets.

6. Now, any model can be trained and evaluated running:
	- MIMIC_train_baselines.py, for LR and Constant models.
	- MIMIC_train_nn.py, for CNN, GRU and CNN-Att.

7. In /notebooks/, you will find:
	- MIMIC_overview.ipynb, where some data analysis from MIMIC-III discharge summaries are provided.
	- MIMIC_analyze_predictions.ipynb, where some additional analyses from the predictions of a model can be seen.


## Analyze CNN_Att model:

We provided our trained CNN_Att weights (SAVE_DIR/cnn_att.h5). To get predictions and metrics for this model:

- Run Steps 1 to 5. 
- Run all cells of notebook notebooks/MIMIC_analyze_predictions.ipynb.

