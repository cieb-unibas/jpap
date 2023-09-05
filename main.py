import sqlite3

import pandas as pd

from jpap import get_company_postings, IPL

# get postings
jpod_conn = sqlite3.connect("/scicore/home/weder/GROUP/Innovation/05_job_adds_data/jpod_test.db")
companies = ["moderna", "roche", "biontech", "novartis"]
df = get_company_postings(con = jpod_conn, companies = companies, institution_name=True)
company_names = df["company_name"].to_list()
postings_texts = df["job_description"].to_list()

# classify postings
classifier = IPL(classifier="pharma")
df["industry"] = classifier(postings = postings_texts, company_names = company_names)

# classify companies
company_industry_labels = df.groupby(["company_name"]).apply(lambda x: x["industry"].value_counts().index[0])
company_industry_labels = {c: i for c in company_industry_labels.index.to_list() for i in company_industry_labels.to_list()}
for company, industry in company_industry_labels.items():
    print(f'"{company} is predicted to be part of the following industry: {industry}"')