Here is an example using the `IPL` pipeline to differentiate between pharmaceutical and non-pharmaceutical companies.

```python
import sys
import os
import sqlite3

sys.path.append(os.getcwd())

from jpap.connect import get_company_postings
from jpap.ipl import IPL
from jpap.preprocessing import subsample_df

# get postings for a set of companies that were not in the training data (max. 5 postings per company)
jpod_conn = sqlite3.connect("/scicore/home/weder/GROUP/Innovation/05_job_adds_data/jpod_test.db")
companies = [
            "sanofi", "incyte", "vertex pharmaceuticals", "abbott laboratories", "baxter", "viatris", # pharma
            "georg fischer", "fresenius", "porsche", "adidas", "sonova", "richemont", "logitech", "ruag",# medtech/manufacturing
            "merrill lynch", "pictet",# banks & insurances
            "astreya", "western digital", "snapchat",# IT
            "utmb health", "hôpital du jura", # health
            "fachhochschule nordwestschweiz fhnw", # research
            "tibits", "grand hotel des bains kempinski st. moritz", "aloft hotels",# hotels & restaurants
            "sbb cff ffs", "20 minuten", "massachusetts department of transportation" # other
            ]
df = get_company_postings(con = jpod_conn, companies = companies, institution_name=True)
df = subsample_df(df=df, group_col="company_name", max_n_per_group=3).reset_index(drop=True)
company_names = df["company_name"].to_list()
postings_texts = df["job_description"].to_list()
assert len(company_names) == len(postings_texts)
print(f'"Predicting the industry association for {len(postings_texts)} postings of {len(set(company_names))} different companies"')

# load pipeline and predict all postings
industry_level = "pharma"
industry_pipeline = IPL(classifier = industry_level)
df["industry"] = industry_pipeline(postings = postings_texts, company_names = company_names)

# majority vote for every companies
company_industry_labels = df.groupby(["company_name"]).apply(lambda x: x["industry"].value_counts().index[0]).to_dict()
for company, industry in company_industry_labels.items():
    print(f'"{company} is predicted to be part of the following industry: {industry}"')
```

Sending this script to the cluster using `ex_scicore.sh` so it can be processed by GPU's **yields the following result:**

```
"Predicting the industry association for 54 postings of 27 different companies"

"Classification is performed at the following level: pharma"

"20 minuten is predicted to be part of the following industry: other"
"abbott laboratories is predicted to be part of the following industry: other"
"adidas is predicted to be part of the following industry: other"
"aloft hotels is predicted to be part of the following industry: other"
"astreya is predicted to be part of the following industry: other"
"baxter is predicted to be part of the following industry: pharmaceutical and life sciences"
"fachhochschule nordwestschweiz fhnw is predicted to be part of the following industry: other"
"fresenius is predicted to be part of the following industry: other"
"georg fischer is predicted to be part of the following industry: other"
"grand hotel des bains kempinski st. moritz is predicted to be part of the following industry: other"
"hôpital du jura is predicted to be part of the following industry: other"
"incyte is predicted to be part of the following industry: pharmaceutical and life sciences"
"logitech is predicted to be part of the following industry: other"
"massachusetts department of transportation is predicted to be part of the following industry: other"
"merrill lynch is predicted to be part of the following industry: other"
"pictet is predicted to be part of the following industry: other"
"porsche is predicted to be part of the following industry: other"
"richemont is predicted to be part of the following industry: other"
"ruag is predicted to be part of the following industry: other"
"sbb cff ffs is predicted to be part of the following industry: other"
"snapchat is predicted to be part of the following industry: pharmaceutical and life sciences"
"sonova is predicted to be part of the following industry: other"
"tibits is predicted to be part of the following industry: other"
"utmb health is predicted to be part of the following industry: other"
"vertex pharmaceuticals is predicted to be part of the following industry: pharmaceutical and life sciences"
"viatris is predicted to be part of the following industry: other"
"western digital is predicted to be part of the following industry: other"
```
