import sqlite3

import pandas as pd

from jpap import get_company_postings, IPL


# get postings
jpod_conn = sqlite3.connect("/scicore/home/weder/GROUP/Innovation/05_job_adds_data/jpod_test.db")
company = "moderna"
postings = get_company_postings(con = jpod_conn, companies= [company], institution_name=True)["job_description"]

# classify all postings
classifier = IPL(classifier="pharma")
industry_predictions = classifier(postings = postings, company_names = [company] * len(postings))

# summarize
df = pd.DataFrame(
    {"postings_text": postings,
     "company": [company] * len(postings),
     "predicted_industry": industry_predictions}
     )
final_label = df["predicted_industry"].value_counts().sort_values().index.to_list()[0]

print(f'"{company} is predicted to be part of the following industry: {final_label}"')