####
# MIMIC Preprocessing
# Merge NOTEEVENTS with DIAGNOSES_ICD tables
# This script creates a table with HADM_IDs linked to unique discharge summaries and a correspondent ICD codes
####

import datasets

mimic = datasets.MIMIC_Dataset()

mimic.preprocess()

mimic.save_preprocessed()


