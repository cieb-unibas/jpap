import sqlite3
import json

import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd

try:
    from .. import preprocessing as pp
    from .. import utils
except:
    from jpap import preprocessing as pp
    from jpap import utils

# configuration
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

# data:

# tokenize the texts:
inputs = tokenizer(list(df["job_description"]), padding = "max_length", truncation = True, return_tensors="pt")
# this is equivalent to running:
# tokens = tokenizer.tokenize(sequence)
# inputs = tokenizer.convert_tokens_to_ids(tokens)
# => und dann sequences mit [101] rsp. [102] verlängern.

# get the model predictions:
outputs = model(**inputs)
outputs
f'Model has {outputs.last_hidden_state.shape[2]} dimension in last hidden layer'

### data -----
def load_labelled(file = "jpap/data/industry_label_companies.json"):
    """
    Loads company names that are labelled to an industry. 
    """
    with open(file, "r", encoding = "UTF-8") as f:
        labelled_companies = json.load(f)
    return labelled_companies

def company_name_pattern_query(patterns):
    """
    Extract a list of companies that match certain patterns.
    
    Parameters:
    ----------
    patterns : list
        A list of patterns to search in the matching variable.
    Returns:
    --------
    str:
        A string in a SQL query format.
    """
    like_statement = utils.sql_like_statement(patterns = patterns)
    
    query_string = """
    SELECT company_name
    FROM position_characteristics
    WHERE (%s)
    """ % like_statement

    return query_string

def get_companies_from_patterns(con, query):
    """
    Extract a list of companies that match a certain pattern in there name.
    """
    company_list = con.execute(query).fetchall()
    company_list = [c[0] for c in company_list]
    return company_list

def create_training_dataset(con):
    """
    Creates a dataset of labelled job postings with respect to industries.
    """
    # get labelled employers
    labelled_employers = load_labelled(file = "jpap/data/industry_label_companies.json")
    # get labelled employer-name patterns and extract employers that match labelled patterns
    patterns = load_labelled(file = "jpap/data/industry_label_patterns.json")
    for i, p in patterns.items():
        pattern_companies = get_companies_from_patterns(
            con = con, 
            query = company_name_pattern_query(patterns = p)
            )
        if not pattern_companies:
            continue
        if i in labelled_employers.keys():
            labelled_employers[i].append(pattern_companies)
        else:
            labelled_employers[i] = pattern_companies
    # extract all (english) postings from these employers:
    employer_postings = pp.get_company_postings(
        con = con, companies = utils.dict_items_to_list(labelled_employers), 
        institution_name = True, language = "eng")
    employer_postings = employer_postings
    # add the industry labels:
    employer_postings["industry"] = employer_postings["company_name"].map(lambda x: [i for i, c in labelled_employers.items() if x in c][0])
    
    return employer_postings[["job_description", "company_name", "industry"]]

if __name__ == "main":
    JPOD_CON = sqlite3.connect("C:/Users/matth/Desktop/jpod_test.db")
    df = create_training_dataset(con = JPOD_CON)
    df.groupby(["industry"]).count()