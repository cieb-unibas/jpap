import random
import os
import sys

from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd

from jpap import industryPipeline as ipl

#### paths / directories
HOME = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
HOME = os.getcwd()
sys.path.append(HOME)

#### extract company descriptions
def load_labelled(n_postings : int = None, home_dir: str = HOME) -> pd.DataFrame:
    """
    Load n postings that are manually labelled to industries.
    """
    df = pd.read_csv(home_dir + "/data/created/industry_train.csv")
    if n_postings:
        random.seed(1)
        out_postings = random.choices(range(len(df)), k = n_postings)
        df = df.iloc[out_postings, :]
    return df

def extract_employer_description(zsc: bool = False) -> pd.DataFrame:
    # load data for training
    df = load_labelled()
    # load and configure description ectractor, extract description by company name
    extractor = ipl.DescExtractor(postings = df["job_description"])
    extractor.by_name(employer_names = df["company_name"])
    # enrich extracted description using zsc
    if zsc:
        classifier = pipeline("zero-shot-classification", "facebook/bart-large-mnli")
        extractor.by_zsc(classifier=classifier, targets = ["who we are", "who this is"])

    df["employer_description"] = [None if x == "" else x for x in extractor.employer_desc]
    return df

df = extract_employer_description().dropna().reset_index(drop=True)[["employer_description", "industry"]]


#### tokenize the input:

#### load BERT-classifier:

#### fine-tune BERT:

#### save the fine-tuned classifier
