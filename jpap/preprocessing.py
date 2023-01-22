import sqlite3
import pandas as pd

def get_postings(con, institution_name = False, html = False, sample_size = False):
    """
    Retrieve postings text from JPOD.
    """
    retrieve_cols = "jp.uniq_id, jp.text_language, jp.job_description"
    join_statement = ""

    if institution_name:
        retrieve_cols += ", pc.company_name"
        join_statement = "LEFT JOIN position_characteristics pc on jp.uniq_id = pc.uniq_id"
    if html:
        retrieve_cols += ", jp.html_job_description"
    if sample_size:
        assert isinstance(sample_size, int)
        limit_condition = "LIMIT %d" % sample_size
    else:
        limit_condition = ""
    
    query = """
    SELECT %s
    FROM (
        SELECT *
        FROM job_postings
        WHERE unique_posting_text == 'yes'
        %s
        ) jp
    %s
    """ % (retrieve_cols, limit_condition, join_statement)

    return pd.read_sql(query, con)
    
class PostingsTranslator():
    def __init__(self, target_language = "en"):
        self.target_language = target_language

    def load_translator():
        "Load a torch model to translate"

    def translate(self):
        "translate the text to english"
