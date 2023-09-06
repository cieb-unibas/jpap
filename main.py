import sqlite3

from jpap.connect import get_company_postings
from jpap.ipl import IPL
from jpap.preprocessing import subsample_df

# get postings for a set of companies (max. 5 postings per company)
jpod_conn = sqlite3.connect("/scicore/home/weder/GROUP/Innovation/05_job_adds_data/jpod_test.db")
companies = [
            "roche", "novartis", "amgen", "bachem", "indorsia", "medartis", "johnson & johnson",
            "alcon", "straumann", "helvetia", "die mobiliar", "bayer", "sanofi", "astrazeneca",
            "sbb", "credit suisse", "google", "holcim", "tibits", "die post",
            "abb", "inselspital", "postfinance", "axpo", "burgerking", "ypsomed",
            "sbb cff ffs", "grand hotel des bains kempinski st. moritz"
            ]
df = get_company_postings(con = jpod_conn, companies = companies, institution_name=True)
df = subsample_df(df=df, group_col="company_name", max_n_per_group=3).reset_index(drop=True)
company_names = df["company_name"].to_list()
postings_texts = df["job_description"].to_list()

# load pipeline and predict all postings
industry_pipeline = IPL(classifier = "pharma")
df["industry"] = industry_pipeline(postings = postings_texts, company_names = company_names)

# majority vote for every companies
company_industry_labels = df.groupby(["company_name"]).apply(lambda x: x["industry"].value_counts().index[0]).to_dict()
for company, industry in company_industry_labels.items():
    print(f'"{company} is predicted to be part of the following industry: {industry}"')