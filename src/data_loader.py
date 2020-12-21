"""
data_loader.py

In this module, the following tasks are performed

- creating vocab dictionary
- creating tag dictionary

"""
import os

DATA_DIR = "./data/"

class DataLoader(object):
    def __init__(self):

        
        # load vocab indices
        vocab_path = os.path.join(DATA_DIR, 


