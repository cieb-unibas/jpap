import os
import sqlite3

import nltk

from jpap import training as jt

# load data:
df = jt.create_training_dataset(con = sqlite3.connect("C:/Users/matth/Desktop/jpod_test.db"), save_dir=False)

# extract the company describing sentences 