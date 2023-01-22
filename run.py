import sqlite3
from jpap import preprocessing as jpp
from transformers import pipeline

#### connect to JPOD and retrieve data
JPOD_CON = sqlite3.connect("C:/Users/matth/Desktop/jpod_test.db")

#### get postings and language information:
postings = jpp.get_postings(con = JPOD_CON, institution_name=True)
postings_lan = {}
for lan in ["eng", "ger", "fre", "ita"]:
    postings_lan[lan] = postings[postings["text_language"] == lan].reset_index().drop("index", axis=1)
    print("%d postings stored for language '%s'" % (len(postings_lan[lan]), lan))

#### choose a posting and tokenize its text to sentences:
postings_lan["eng"]["company_name"][180:200]
n = 198
company = postings_lan["eng"]["company_name"][n]
text = postings_lan["eng"]["job_description"][n]

#### A) ASSIGN POSTINGS TO SECTORS-------------------
from jpap import industry_pipeline as ipl

# load a pre-trained zero-shot-learning classifier from huggingface:
classifier = pipeline("zero-shot-classification", "facebook/bart-large-mnli")
sectors = [
    "research and academia", "schools and teaching", "financial and insurance industry", 
    "manufacturing industry", "pharmaceutical and life sciences industry", 
    "politics and public administration", "hotels and restaurant industry", 
    "retail, wholesale and stores industry", "electricity and energy industry",
    "medicine and hospitals industry", "newspapers, television and media industry", 
    "information, communication and telecommunication industry"
    ]
res = ipl.posting_to_industry(
    text = text, classifier = classifier, sectors = sectors, 
    company_name = company
    # , retrieve_probas=True
    #, select_sentence_by="company_name"
    )
res["company"], res["industry"]

# check the selected sentences
import nltk
ipl.retrieve_company_description(nltk.sent_tokenize(text),classifier=classifier)

# => experiment with few-shot-classifiers instead... use big companies/organizations where we know the ground truth and label them for training.