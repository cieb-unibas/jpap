import sqlite3
from jpap import get_company_postings, IPL, subsample_df

# get postings for a set of companies (max 5 postings per company)
jpod_conn = sqlite3.connect("/scicore/home/weder/GROUP/Innovation/05_job_adds_data/jpod_test.db")
companies = [
            "roche", "novartis", "sbb", "credit suisse", "google", "holcim", 
            "abb", "inselspital", "postfinance", "axpo", "universität bern",
            "sbb cff ffs", "grand hotel des bains kempinski st. moritz"
            ]
df = get_company_postings(con = jpod_conn, companies = companies, institution_name=True)
df = subsample_df(df=df, group_col="company_name", max_n_per_group=3).reset_index(drop=True)
company_names = df["company_name"].to_list()
postings_texts = df["job_description"].to_list()

# inference
industry_pipeline = IPL(classifier="pharma")
df["industry"] = industry_pipeline(postings = postings_texts, company_names = company_names)

# majority vote for companies
company_industry_labels = df.groupby(["company_name"]).apply(lambda x: x["industry"].value_counts().index[0])
company_industry_labels = {c: i for c in company_industry_labels.index.to_list() for i in company_industry_labels.to_list()}
for company, industry in company_industry_labels.items():
    print(f'"{company} is predicted to be part of the following industry: {industry}"')