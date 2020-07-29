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

## Edit this file with your desired values

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
