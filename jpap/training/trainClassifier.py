import os

from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd
import nltk

from jpap import industry as ipl

#### data
n_postings = 2
df = pd.read_csv("data/created/industry_train.csv").iloc[:n_postings,:]

#### only keep relevant sentences:

# load a pre-trained zero-shot-learning classifier from huggingface:
classifier = pipeline("zero-shot-classification", "facebook/bart-large-mnli")
company_description = []
for p in df["job_description"]:
    desc = ipl.retrieve_company_description(nltk.sent_tokenize(p), classifier=classifier)
    # TO DO: MAYBE CHANGE THE ZSC TO HAVE THE FOLLOWING LABELS: 'WHO ARE WE', 'WHO IS THIS', 'OTHER' TO FILTER OUT TEAM DESCRIPTION
    company_description.append(desc)
company_description = [" ".join(c) for c in company_description]
df["company_description"] = company_description

#### model configuration --------------------
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

#### tokenization ------------------
def tokenization(x, tokenizer):
    x_tokenized = tokenizer(x, padding = "max_length", truncation = True, return_tensors="pt")
    # this is equivalent to running:
    # tokens = tokenizer.tokenize(sequence)
    # inputs = tokenizer.convert_tokens_to_ids(tokens)
    # => und dann sequences mit [101] rsp. [102] verlängern (je nach model).
    return x_tokenized 

model_inputs = [tokenization(c, tokenizer=tokenizer) for c in list(df["company_description"])]



# get the model predictions:
outputs = model(**model_inputs)
outputs
f'Model has {outputs.last_hidden_state.shape[2]} dimension in last hidden layer'