#### 
# Constants

# Edit this file with your directories and desired values
####

from pathlib import Path
REPO_PATH = str(Path(__file__).resolve().parents[0])

# Word2Vec embedding word-vector dimension
W2V_SIZE = 300

# Fixed length to pad/truncate input samples
MAX_LENGTH = 2000



# Directories

# Path for MIMI-III tables
DATA_DIR = REPO_PATH + '/data/'

# Path to save Word2Vec embeddings 
W2V_DIR = REPO_PATH + '/models/w2v/'

# Path to save trained models
SAVE_DIR = REPO_PATH + '/models/'
