import sqlite3
import json

import torch
from transformers import AutoTokenizer, AutoModel

try:
    from .. import preprocessing as pp
except:
    from jpap import preprocessing as pp

# configuration
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

# data:
JPOD_CON = sqlite3.connect("C:/Users/matth/Desktop/jpod_test.db")
postings = pp.get_postings(con = JPOD_CON, institution_name=True, language="eng")
postings_lan = {}
for lan in ["eng", "ger", "fre", "ita"]:
    postings_lan[lan] = postings[postings["text_language"] == lan].reset_index(drop=True)
    print("%d postings stored for language '%s'" % (len(postings_lan[lan]), lan))
postings_lan["eng"]["company_name"][250:300]
idx = [298, 285, 265, 266]
df = postings_lan["eng"].iloc[idx,:][["job_description", "company_name"]].reset_index(drop = True)
for t in df["job_description"]:
    print(f'Text length: {len(t.split(" "))}')

# tokenize the texts:
inputs = tokenizer(list(df["job_description"]), padding = "max_length", truncation = True, return_tensors="pt")
# this is equivalent to running:
# tokens = tokenizer.tokenize(sequence)
# inputs = tokenizer.convert_tokens_to_ids(tokens)
# => und dann sequences mit [101] rsp. [102] verlängern.

# get the model predictions:
outputs = model(**inputs)
outputs
f'Model has {outputs.last_hidden_state.shape[2]} dimension in last hidden layer'







# --------------------------

def load_training_dataset():
    """
    Load the labelled postings
    """
    # with open("../data/industry_company_labels.json") as f:
    #     train_dat = json.load(f)
    with open("jpap/data/industry_company_labels.json", "r", encoding = "UTF-8") as f:
        train_dat = json.load(f)
    companies = []
    for company_list in train_dat.values():
        companies += company_list
    
    # load and label postings data
    postings = pp.get_company_postings(con = JPOD_CON, companies = companies, institution_name=True, language="eng")[["job_description", "company_name"]]
    company_postings = pd.DataFrame(list(set(list(postings["company_name"]))), columns = ["company_name"])
    industry_labels = []
    for c in company_postings["company_name"]:
        industry_labels += [i for i, j in train_dat.items() if c in j]
    company_postings["industry"] = industry_labels
    postings = postings.merge(company_postings, on = "company_name")

    return postings

df = load_training_dataset()
df.groupby(["industry"]).count()