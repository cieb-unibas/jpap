import sqlite3
import pandas as pd

def get_postings(con, institution_name: bool = False, html: bool = False, sample_size = False, language: str = None):
    """
    Retrieve postings text from JPOD.
    """
    retrieve_cols = "jp.uniq_id, jp.text_language, jp.job_description"
    language_condition = ""
    limit_condition = ""
    join_statement = ""


    if institution_name:
        retrieve_cols += ", pc.company_name"
        join_statement = "LEFT JOIN position_characteristics pc on jp.uniq_id = pc.uniq_id"
    if html:
        retrieve_cols += ", jp.html_job_description"
    if language:
        language_condition = "AND text_language == '%s'" %language
    if sample_size:
        assert isinstance(sample_size, int)
        limit_condition = "LIMIT %d" % sample_size
    
    query = """
    SELECT %s
    FROM (
        SELECT *
        FROM job_postings
        WHERE unique_posting_text == 'yes' %s
        %s
        ) jp
    %s
    """ % (retrieve_cols, language_condition, limit_condition, join_statement)

    return pd.read_sql(query, con)

def get_company_postings(con, companies, institution_name: bool = False, html: bool = False, language: str = None):
    retrieve_cols = "jp.uniq_id, jp.text_language, jp.job_description"
    language_condition = ""
    join_statement = ""

    if institution_name:
        retrieve_cols += ", pc.company_name"
        join_statement = "LEFT JOIN position_characteristics pc on jp.uniq_id = pc.uniq_id"
    if html:
        retrieve_cols += ", jp.html_job_description"
    if language:
        language_condition = "AND text_language == '%s'" %language
    
    query = """
    SELECT %s
    FROM (
        SELECT *
        FROM job_postings
        WHERE unique_posting_text == 'yes'
        AND uniq_id IN (SELECT uniq_id FROM position_characteristics WHERE company_name IN (%s))
        %s
        ) jp
    %s
    """ % (retrieve_cols, str(companies)[1:-1], language_condition, join_statement)

    return pd.read_sql(query, con)
    
class PostingsTranslator():
    def __init__(self, target_language = "en"):
        self.target_language = target_language

    def load_translator():
        "Load a torch model to translate"

    def translate(self):
        "translate the text to english"
