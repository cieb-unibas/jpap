import random
import os
import sys

from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd

from scripts.itp.DescExtractor import DescExtractor

# paths / directories
HOME = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
HOME = os.getcwd()
sys.path.append(HOME)

def load_labelled(n_postings : int = None, home_dir: str = HOME):
    """
    Load n labelled postings.
    """
    df = pd.read_csv(home_dir + "/data/created/industry_train.csv")
    if n_postings:
        random.seed(1)
        out_postings = random.choices(range(len(df)), k = n_postings)
        df = df.iloc[out_postings, :]
    return df

def extract_employer_description(zsc = False):

    df = load_labelled()
    Extractor = DescExtractor(postings = df["job_description"])

    Extractor.by_name(employer_names = df["company_name"])

    if zsc:
        classifier = pipeline("zero-shot-classification", "facebook/bart-large-mnli")
        Extractor.by_zsc(classifier=classifier, targets = ["who we are", "who this is"])
    
    return df["uniq_id"], Extractor.employer_names, Extractor.employer_desc
df = pd.DataFrame()
df["uniq_id"], df["employer"], df["desc"] = extract_employer_description()
