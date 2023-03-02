import json
import os
import sys
import sqlite3

HOME = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
sys.path.append(HOME)
import jpap

try:
    JPOD_CON = sqlite3.connect("C:/Users/matth/Desktop/jpod_test.db")
except:
    JPOD_CON = sqlite3.connect("C:/Users/nigmat01/Desktop/jpod_test.db")


def storeat(home_dir = HOME, file_name = "industry_train.csv"):
    dataDir = home_dir + "/data/created/" + file_name
    return dataDir

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
    label_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, "data", "raw"))
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
    like_statement = jpap.utils.sql_like_statement(patterns = patterns)
    
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

def create_training_dataset(con, save = False, peak = False):
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

    # extract all (english) postings from these employers:
    employer_postings = jpap.preprocessing.get_company_postings(
        con = con, companies = jpap.utils.dict_items_to_list(labelled_employers), 
        institution_name = True, language = "eng")
    employer_postings = employer_postings

    # add the industry labels:
    employer_postings["industry"] = employer_postings["company_name"].map(lambda x: [i for i, c in labelled_employers.items() if x in c][0])
    print("********** Extracted %d postings from the database.**********" % len(employer_postings))

    # save and return
    if save:
        savefile = storeat()
        employer_postings.to_csv(savefile, index = False)
        print("Training dataset saved to %s" % savefile)
    if peak:
        return print(employer_postings[["job_description", "company_name", "industry"]].head())


if __name__ == "__main__":
    create_training_dataset(con = JPOD_CON, save = True, peak=True)
