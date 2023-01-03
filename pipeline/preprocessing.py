import sqlite3
import pandas as pd

class DataRetriever():
    def __init__(self, db_path):
        self.db_path = db_path
        self.con = sqlite3.connect(db_path)

    def get_postings(self, html = False, sample_size = False):
        """
        Retrieve postings text from JPOD.
        """
        if html:
            retrieve_col = "html_job_description"
        else:
            retrieve_col = "job_description"

        if sample_size:
            assert isinstance(sample_size, int)
            limit_condition = "LIMIT %d" % sample_size
        else:
            limit_condition = ""
        
        query = """
        SELECT uniq_id, text_language, %s
        FROM job_postings
        WHERE unique_posting_text == 'yes'
        %s
        """ % (retrieve_col, limit_condition)

        postings = pd.read_sql(query, self.con)
        
        return postings

class PostingsTranslator():
    def __init__(self, target_language = "en"):
        self.target_language = target_language

    def load_translator():
        "Load a torch model to translate"

    def translate(self):
        "translate the text to english"
