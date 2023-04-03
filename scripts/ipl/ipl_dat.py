import json
import os
import sys
import sqlite3

from transformers import pipeline

HOME = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
#HOME = os.getcwd()
sys.path.append(HOME)

import jpap

try:
    JPOD_CON = sqlite3.connect("C:/Users/matth/Desktop/jpod_test.db")
except:
    JPOD_CON = sqlite3.connect("C:/Users/nigmat01/Desktop/jpod_test.db")


def _storeat(home_dir = HOME, file_name = "industry_train.csv"):
    dataDir = home_dir + "/data/created/" + file_name
    return dataDir

def _load_labels(label_type = "companies", home_dir = HOME):
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
    label_path = os.path.abspath(os.path.join(home_dir, "data", "raw"))
    file = os.path.join(label_path, "industry_label_" + label_type + ".json")
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
    Extract a list of employera that match a certain pattern in their name.

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

def _extract_employer_description(df, zsc: bool = False):
    """
    Extract sentences of a posting that describe the employer based on
    employer name and/or zero-shot sentence classifier.
    """
    # extract description by company name
    extractor = jpap.DescExtractor(postings = df["job_description"])
    extractor.by_name(employer_names = df["company_name"])
    # enrich extracted description using zsc
    if zsc:
        classifier = pipeline("zero-shot-classification", "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")
        desc_targets = ["who we are", "who this is", "industry or sector"]
        extractor.by_zsc(classifier=classifier, targets = desc_targets)
    df["employer_description"] = [None if x == "" else x for x in extractor.employer_desc]
    return df

def create_training_dataset(con, save = False, peak = False, use_zsc = False):
    """
    Creates a dataset of labelled job postings with respect to industries.
    
    Paramaters:
    ----------
    con: sqlite.Connection
        A connection to the sqlite database        
    """
    labelled_employers = _load_labels(label_type = "companies")

    # extract labelled employer-name patterns and add employers that match these labelled name patterns
    patterns = _load_labels(label_type = "patterns")
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

    # extract all postings from these employers:
    employer_postings = jpap.get_company_postings(
        con = con, companies = jpap.dict_items_to_list(labelled_employers), 
        institution_name = True)
    employer_postings = employer_postings   

    # add the industry labels:
    employer_postings["industry"] = employer_postings["company_name"].map(lambda x: [i for i, c in labelled_employers.items() if x in c][0])

    # extract employer description:
    if use_zsc:
        print("Extracting employer descriptions by name-searches AND zero-shot sentence classifier.")
    else:
        print("Extracting employer descriptions by name-searches only.")
    employer_descriptions = _extract_employer_description(df = employer_postings, zsc = use_zsc)
    employer_descriptions = employer_descriptions.dropna().reset_index(drop=True)[["employer_description", "industry"]]
    print("********** Extracted employer descriptions for %d postings from the database.**********" % len(employer_descriptions))

    # save and return
    if save:
        savefile = _storeat()
        employer_descriptions.to_csv(savefile, index = True)
        print("Training dataset saved to %s" % savefile)
    if peak:
        return print(employer_descriptions.head())


if __name__ == "__main__":
    create_training_dataset(con = JPOD_CON, save = True, peak=True, use_zsc = True)
