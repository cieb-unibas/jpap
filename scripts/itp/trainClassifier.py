import os

from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd
import nltk

from scripts.itp import industry as ipl

#### data
n_postings = 2
df = pd.read_csv("data/created/industry_train.csv").iloc[:n_postings,:]

#### only keep relevant sentences:
# => reframe the pipeline:
# 1) retrieve the sentences where the company's name is stated.
# 2) take all other sentences and tokenize them.
#   a) maybe remove stopwords
#   b) maybe use another zsc for the company detection
# 3) use the zsc to add additional sentences without the company name
# 4) bind everything together


def retrieve_companyname_sentences(sentences, company_name, verbose = False):
    company_sentences = [s for s in sentences if company_name in s.lower()]
    if verbose:
        print("Classified %d sentences as relevant for company description" % len(company_sentences))
    return company_sentences
 
def retrieve_companydescribing_sentences(sentences, classifier, targets, verbose = False):
    labels = targets + ["other"]
    # 
    company_sentences = [classifier(s, candidate_labels=labels) for s in sentences]
    company_sentences = [s["sequence"] for s in company_sentences if s["labels"][0] in targets]
    if verbose:
        print("Classified %d sentences as relevant for company description using zero-shot-learning" % len(company_sentences))
    return company_sentences

retrieve_companyname_sentences(sentences=nltk.sent_tokenize(df["job_description"][1]), company_name="credit suisse")

# load a pre-trained zero-shot-learning classifier from huggingface:
classifier = pipeline("zero-shot-classification", "facebook/bart-large-mnli")
targets = ["who we are", "who this is"]
company_description = []
for p in df["job_description"]:
    desc = retrieve_companydescribing_sentences(
        sentences = nltk.sent_tokenize(p),
        classifier=classifier,
        targets=targets)
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