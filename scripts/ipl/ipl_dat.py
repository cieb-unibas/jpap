import json
import os
import sys
import sqlite3

import pandas as pd
from transformers import pipeline

print("Current directory is: ", os.getcwd())
sys.path.append(os.getcwd())

import jpap

# define functions
def _load_labels(data_dir, label_type = "companies"):
    """
    Loads company names or company name patterns that are labelled to industries.

    Paramater:
    ---------
    type: str
        A string indicating whether labelled company names or labelled company patterns are loaded.
        Must be either `companies` or `patterns`.

    Returns:
    -------
    dict:
        A dictionary with keys indicating industry labels and values representing company names or company name patterns.
    """
    assert label_type in ["companies", "patterns"], "`type` must be one of `companies` or `patterns`."
    file = os.path.join(data_dir, "industry_label_" + label_type + ".json")
    with open(file, "r", encoding = "UTF-8") as f:
        labels = json.load(f)
    return labels

def _name_pattern_query(patterns):
    """
    Extract a list of employers that match certain patterns.
    
    Parameters:
    ----------
    patterns : list
        A list of patterns to search in the matching variable.
    Returns:
    --------
    str:
        A string in a SQL query format.
    """
    like_statement = jpap.sql_like_statement(patterns = patterns)
    
    query_string = """
    SELECT company_name
    FROM position_characteristics
    WHERE (%s)
    """ % like_statement

    return query_string

def _employers_from_patterns(con, query):
    """
    Extract a list of employers from JPOD that match a certain pattern in their name.

    Parameters:
    ----------
    con: sqlite.Connection
        A connection to the sqlite database
    query: str
        A string in a SQL query format.
    
    Returns:
    --------
    list:
        A list of employer names that are present in the database.
    """
    employer_list = con.execute(query).fetchall()
    employer_list = [c[0] for c in employer_list]
    return employer_list

def _extract_employer_description(df, path_to_models, zsc = False, 
                                  model = "multilingual-MiniLMv2-L6-mnli-xnli/",
                                  ):
    """
    Extract sentences of a posting that describe the employer based on
    employer name and/or zero-shot sentence classifier.
    """
    # init the jpap.DescExtractor class
    extractor = jpap.DescExtractor(postings = df["job_description"])
    
    # extract description by company name
    extractor.sentences_by_name(employer_names = df["company_name"])
    
    # enrich extracted description using zsc
    if zsc:
        if path_to_models:
            model_path = os.path.join(path_to_models, model)
            classifier = pipeline(task = "zero-shot-classification", model = model_path)
        else:
            classifier = pipeline("zero-shot-classification", "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")
        desc_targets = ["who we are", "who this is", "industry or sector"]
        extractor.sentences_by_zsc(classifier=classifier, targets = desc_targets)
    
    df["employer_description"] = [None if x == "" else x for x in extractor.employer_desc]
    
    return df
    
def create_training_dataset(con, data_dir, model_dir = None, save_as = None, peak = False, use_zsc = False):
    """
    Creates a dataset of labelled job postings with respect to industries.
    
    Paramaters:
    ----------
    con: sqlite.Connection
        A connection to the sqlite database
    data_dir: str
        A string indicating the location of industry-labelled job postings
    model_dir: str
        A string indicating the location of classification models that are loaded from disk. Default is None
    save_as: str
        A string indicating the path under which the resulting data is to be saved.
    peak: bool
        A flag indicating if a peak of the resulting data should be printed
    use_zsc: bool
        A flag indicating if a zero-shot-classifier should be used to extract relevant sentences for company
        descriptions. Default is False indicating that extraction is based on company names only.            
    """
    # load labelled employers
    labelled_employers = _load_labels(data_dir = data_dir, label_type = "companies")

    # extract labelled employer-name patterns and add employers that match labelled name patterns
    patterns = _load_labels(data_dir = data_dir, label_type = "patterns")
    for i, p in patterns.items():
        pattern_companies = _employers_from_patterns(
            con = con, 
            query = _name_pattern_query(patterns = p)
            )
        if not pattern_companies:
            continue
        if i in labelled_employers.keys():
            labelled_employers[i].append(pattern_companies)
        else:
            labelled_employers[i] = pattern_companies

    # extract all postings from these employers, add their industry labels and define sample size:
    employer_postings = jpap.get_company_postings(
        con = con, companies = jpap.dict_items_to_list(labelled_employers), 
        institution_name = True)
    
    industry_labels = employer_postings["company_name"].map(lambda x: [i for i, c in labelled_employers.items() if x in c])
    employer_postings["industry"] = [le[0] if len(le) > 0 else None for le in industry_labels]
    employer_postings = employer_postings.dropna().reset_index(drop=True)
    employer_postings = jpap.subsample_df(df = employer_postings, group_col= "industry", max_n_per_group = 500)
    employer_postings = employer_postings.dropna().reset_index(drop=True)

    # extract employer description:
    if use_zsc:
        print("Extracting employer descriptions by name-searches AND zero-shot sentence classifier.")
    else:
        print("Extracting employer descriptions by name-searches only.")
    employer_descriptions = _extract_employer_description(
        df = employer_postings, 
        zsc = use_zsc, path_to_models = model_dir
        )
    employer_descriptions = employer_descriptions.dropna().reset_index(drop=True)[["employer_description", "company_name","industry"]]
    print("********** Extracted employer descriptions for %d postings from the database.**********" % len(employer_descriptions))

    # save and return
    if save_as:
        employer_descriptions.to_csv(save_as, index = False)
        print("Training dataset saved to %s" % save_as)
    if peak:
        print(employer_descriptions.head())

if __name__ == "__main__":

    # set directories
    BASE_DIR = "/scicore/home/weder/GROUP/Innovation/05_job_adds_data/"
    DB_VERSION = "jpod.db"
    DATA_DIR = os.path.join(BASE_DIR, "jpap/data/raw/")
    MODEL_DIR = os.path.join(BASE_DIR, "hf_models/")
    SAVE_AS = os.path.join(BASE_DIR, "augmentation_data/industry_train.csv")
    
    # databse connection
    JPOD_CON = sqlite3.connect(os.path.join(BASE_DIR, DB_VERSION))
    
    # create dataset
    create_training_dataset(
        con = JPOD_CON, save_as = SAVE_AS, data_dir = DATA_DIR,
        model_dir = MODEL_DIR, peak=True, use_zsc = True 
        )