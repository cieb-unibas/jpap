import random
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

from jpap.preprocessing import encode_labels

#### paths / directories
def set_paths():
    try:
        HOME = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
    except:
        HOME = os.getcwd()
    sys.path.append(HOME)
    return HOME

#### extract company descriptions
def load_labelled(n_postings : int = None, home_dir: str = set_paths()) -> pd.DataFrame:
    """
    Load n postings' employer descriptions that were manually labelled to industries.
    """
    df = pd.read_csv(home_dir + "/data/created/industry_train.csv")
    if n_postings:
        random.seed(1)
        out_postings = random.choices(range(len(df)), k = n_postings)
        df = df.iloc[out_postings, :]
        df = df.reset_index(drop=True)
    return df

#### extract company descriptions
def load_xlm_pretrained(path_to_model = None):
    if path_to_model:
        tokenizer = AutoTokenizer.from_pretrained(path_to_model)
        model = AutoModelForSequenceClassification.from_pretrained(path_to_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base")
    return tokenizer, model



#### load xlm-roberta-base-classifier:
tokenizer, model = load_xlm_pretrained()

#### load and tokenize the input and label the targets
df = load_labelled()

# labels
labels, label_dict = encode_labels(inputs = df["industry"], return_label_dict = True)
labels = torch.tensor(labels, dtype=torch.int32)
# features
inputs = tokenizer(
    df["employer_description"].to_list(), return_tensors="pt", 
    truncation=True, max_length=128, padding=True
    )
# => does not seem right..

#### split data to training and testing sets
df.groupby(["industry"])["industry"].count().sort_values(ascending=False)

#### train/fine-tune classifier for industry classification

#### evaluate the fine-tuned classifier:

#### save the fine-tuned classifier
