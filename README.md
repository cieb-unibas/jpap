# JPAP
The idea of JPAP ('job postings analyses pipelines') is to create different pipelines for in-depth analysis of job postings contained in the CIEB Job Postings Database (JPOD).

## Installation

`cd` into this repository, setup a virtual environment via and install dependencies:

Using conda
```bash
conda create --name jpap python=3.9
conda install -r requirements.txt
```

Using pip
```bash
python venv jpap-venv
pip install --upgrade pip
pip install -r requirements.txt
```

#### Troubleshooting
If you encounter problems installing the dependencies for JPAP, check for GPU support on your machine and setup `PyTorch` manually according to the [developer website](https://pytorch.org/). Consider doing the same  for the `transformers` library as specified by [Huggingface](https://huggingface.co/docs/transformers/installation):


## Job Postings Industry Pipeline (IPL)

### How?
- IPL **leverages pre-trained language models** for feature extraction and classification. Feature extraction is based on the `multilingual-MiniLMv2-L6-mnli-xnli` model, a version of `XLM-RoBERTa-large` finetuned on the `XNLI` and `MNLI` datasets for zero-shot-classification. 

- IPL then follows a **weak supervision** approach with **labelling-fuctions** (LF) to label a subset of postings to industries. It does this based on companies, for which the ground truth is known (e.g. roche is from the pharmaceutical sector) as well as on certain patterns in the company name that can be unambigously assigned to specific industries (e.g. 'university', 'hospital' or 'restaurant').

- Since postings of the same companies can be very similar, **sampling** procedure is an issue because of potential data leakage when splitting training and test set. For example, a job posting of Roche could land in the training and another one in the testing set even though they are (practically) identical. For the IPL training the dataset was does split by employers so that all postings from a certain employer can either be in the test or in the validation set. Furthermore, company names are blinded for the classifier so that it does not classify postings to industries based on company names.

- The IPL classifier is based on **transfer learning** as it is a fine-tuned version of the `XLM-RoBERTa` model, which in itself is a multilingual version of `RoBERTa` model that has been pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages by researchers at Facebook AI. 

### Example
Here is an example using the `IPL` pipeline to differentiate between pharmaceutical and non-pharmaceutical companies. 

```python
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
```

Sending this script to the cluster using `main_scicore.sh` so it can be processed by GPU's **yields the following result:**

"abb is predicted to be part of the following industry: other"
**"astrazeneca is predicted to be part of the following industry: pharmaceutical and life sciences"**
"axpo is predicted to be part of the following industry: other"
"credit suisse is predicted to be part of the following industry: other"
"die mobiliar is predicted to be part of the following industry: other"
"die post is predicted to be part of the following industry: other"
"google is predicted to be part of the following industry: other"
"grand hotel des bains kempinski st. moritz is predicted to be part of the following industry: other"
"helvetia is predicted to be part of the following industry: other"
"inselspital is predicted to be part of the following industry: other"
**"johnson & johnson is predicted to be part of the following industry: pharmaceutical and life sciences"**
**"novartis is predicted to be part of the following industry: pharmaceutical and life sciences"**
"postfinance is predicted to be part of the following industry: other"
**"roche is predicted to be part of the following industry: pharmaceutical and life sciences"**
"sbb cff ffs is predicted to be part of the following industry: other"
"tibits is predicted to be part of the following industry: other"

## Jop Postings Translation Pipeline (TPL)
Multilingual translations for job postings.

- Work in Progress
- Consider using a translation pipeline using fine-tuned encoder-decoder architecture: Check the following [paper](https://arxiv.org/pdf/2010.11125.pdf), [code][https://github.com/kadirnar/Multilingual-Translation] and [model][https://huggingface.co/facebook/m2m100_418M] from Facebook AI.
