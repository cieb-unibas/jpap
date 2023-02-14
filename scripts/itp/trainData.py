import sqlite3
import json
import os

from jpap import preprocessing as pp
from jpap import utils

HERE = os.path.abspath('C:/Users/matth/Documents/github_repos/jpap/jpap/training')
#save_dir = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, "data", "created", "industry_train.csv"))

### data -----
def _load_labels(label_type = "companies"):
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
#    label_path = os.path.abspath(os.path.join(__file__, os.pardir, "data"))
    label_path = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, "data", "raw"))
    file = os.path.join(label_path, "industry_label_" + label_type + ".json")
    with open(file, "r", encoding = "UTF-8") as f:
        labels = json.load(f)
    return labels

def _company_name_pattern_query(patterns):
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

def _get_companies_from_patterns(con, query):
    """
    Extract a list of companies that match a certain pattern in their name.

    Parameters:
    ----------
    con: sqlite.Connection
        A connection to the sqlite database
    query: str
        A string in a SQL query format.
    
    Returns:
    --------
    list:
        A list of company names that are present in the database.
    """
    company_list = con.execute(query).fetchall()
    company_list = [c[0] for c in company_list]
    return company_list

def create_training_dataset(con, save_dir = False):
    """
    Creates a dataset of labelled job postings with respect to industries.
    
    Paramaters:
    ----------
    con: sqlite.Connection
        A connection to the sqlite database        
    """
    # get labelled employers
    labelled_employers = _load_labels(label_type = "companies")
    # get labelled employer-name patterns and extract employers that match labelled patterns
    patterns = _load_labels(label_type = "patterns")
    for i, p in patterns.items():
        pattern_companies = _get_companies_from_patterns(
            con = con, 
            query = _company_name_pattern_query(patterns = p)
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
    print("********** Extracted %d postings from the database.**********" % len(employer_postings))
    # save and return
    if save_dir:
        employer_postings.to_csv(save_dir, index = False)
        print("Training dataset saved to %s" % save_dir)
    return employer_postings[["job_description", "company_name", "industry"]]

if __name__ == "main":
    JPOD_CON = sqlite3.connect("C:/Users/matth/Desktop/jpod_test.db")
    create_training_dataset(con = JPOD_CON, save_dir = save_dir).head()
    