import random
import os
import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

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

#### split data to training and testing sets
df = load_labelled()
df.groupby(["industry"])["industry"].count().sort_values(ascending=False)



#### tokenize the input and label the targets

#### load xlm-roberta-base-classifier:

#### train/fine-tune classifier for industry classification

#### evaluate the fine-tuned classifier:

#### save the fine-tuned classifier
