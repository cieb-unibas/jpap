import random
import os
import sys

import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd
import nltk

from .DescExtractor import DescExtractor

HOME = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
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









# ------------------------------------------ OLD ---------------------


def name_sentences(company_names, postings):
    """
    Extract company describing sentences by employer name.
    """
    assert isinstance(company_names, list)
    assert isinstance(postings, list)
    assert len(company_names) == len(postings)
    description_list = []
    for c, p in zip(company_names, postings):
        sentences = nltk.sent_tokenize(p)
        company_sentences = [s for s in sentences if c in s.lower()]
        if company_sentences:
            desc = " ".join(company_sentences)
        else:
            desc = ""
        description_list.append(desc) # test this here... it should give an empty string
    return description_list

def zsc_input_sentences(postings, labelled_sentences, max_sentences = None):
    """
    Remove already labelled sentences from a text and prepare as tokenized input for a zero-shot-classifier.
    """
    assert isinstance(labelled_sentences, list)
    assert isinstance(postings, list)
    assert len(postings) == len(labelled_sentences)
    zsc_in = []
    for p, ls in zip(postings, labelled_sentences):
        if ls:
            zsc_in_c = [s for s in nltk.sent_tokenize(p) if s not in ls]
            if max_sentences:
                zsc_in_c = zsc_in_c[:max_sentences]
            zsc_in.append(zsc_in_c)
        else: # test this if it works
            zsc_in_c = nltk.sent_tokenize(p)
            if max_sentences:
                zsc_in_c = zsc_in_c[:max_sentences]
            zsc_in.append(zsc_in_c)
    return zsc_in
 
def zsc_sentences(tokenized_postings, classifier, targets):
    """
    Extract company describing sentences by using a zero-shot-classifier.
    """
    labels = targets + ["other"]
    description_list = []
    for p in tokenized_postings:
        company_sentences = [classifier(s, candidate_labels=labels) for s in p]
        company_sentences = [s["sequence"] for s in company_sentences if s["labels"][0] in targets]
        if company_sentences:
            desc = " ".join(company_sentences)
        else:
            desc = ""
        description_list.append(desc) # test this here... it should give an empty string
    return description_list

N = 4
df = load_labelled(N)

# extract by name
name_desc = name_sentences(
    company_names = list(df["company_name"]),
    postings = list(df["job_description"])
    )

# extract by zero-shot-classification 
zsc_desc = zsc_sentences(
    tokenized_postings = zsc_input_sentences(postings = list(df["job_description"]), labelled_sentences = name_desc), 
    classifier = pipeline("zero-shot-classification", "facebook/bart-large-mnli"), 
    targets = ["who we are", "who this is"]
    )

# bind all extracted sentences together and add to dataframe
company_description = []
for n, z in zip(name_desc, zsc_desc):
    desc = n[0] + z[0]
    company_description.append(desc)
df["company_description"] = company_description









#-------------------------------
# load a pre-trained zero-shot-learning classifier from huggingface:
classifier = pipeline("zero-shot-classification", "facebook/bart-large-mnli")
targets = ["who we are", "who this is"]
company_description = []
for p in df["job_description"]:
    desc = retrieve_companydescribing_sentences(
        sentences = nltk.sent_tokenize(p),
        classifier=classifier,
        targets=targets)
    company_description.append(desc)
company_description = [" ".join(c) for c in company_description]
df["company_description"] = company_description

#### model configuration --------------------
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

#### tokenization ------------------
def tokenization(x, tokenizer):
    x_tokenized = tokenizer(x, padding = "max_length", truncation = True, return_tensors="pt")
    # this is equivalent to running:
    # tokens = tokenizer.tokenize(sequence)
    # inputs = tokenizer.convert_tokens_to_ids(tokens)
    # => und dann sequences mit [101] rsp. [102] verlängern (je nach model).
    return x_tokenized 

model_inputs = [tokenization(c, tokenizer=tokenizer) for c in list(df["company_description"])]



# get the model predictions:
outputs = model(**model_inputs)
outputs
f'Model has {outputs.last_hidden_state.shape[2]} dimension in last hidden layer'