import sqlite3

import torch
from transformers import pipeline

from jpap import preprocessing as jpp
from jpap.tpl import ENDetectionModel

#### connect to JPOD and retrieve data
JPOD_CON = sqlite3.connect("C:/Users/matth/Desktop/jpod_test.db")

#### get postings and language information:
postings = jpp.get_postings(con = JPOD_CON, institution_name=True)
postings_lan = {}
for lan in ["eng", "ger", "fre", "ita"]:
    postings_lan[lan] = postings[postings["text_language"] == lan].reset_index().drop("index", axis=1)
    print("%d postings stored for language '%s'" % (len(postings_lan[lan]), lan))

#### choose a posting and tokenize its text to sentences:
postings_lan["eng"]["company_name"][150:200]
n = 133
company = postings_lan["eng"]["company_name"][n]
text = postings_lan["eng"]["job_description"][n]

#### detect english language
from jpap.tpl.endetect import ENDetectionModel, ENDetectTokenizer
tokenizer = ENDetectTokenizer().load_vocabulary()
model = ENDetectionModel().load_state_dict()
n = 27
text = postings_lan["ita"]["job_description"][n]
tokenized_text = tokenizer.tokenize(text = text, sequence_length=50, padding_idx=0)
tokenized_text = torch.tensor([tokenized_text], dtype=torch.int32)
print("The postings text is in english: ", (model(tokenized_text)[:, 0] > 0.5).item())

#### A) ASSIGN POSTINGS TO SECTORS-------------------
from scripts.itp import industry as ipl

# load a pre-trained zero-shot-learning classifier from huggingface:
classifier = pipeline("zero-shot-classification", "facebook/bart-large-mnli")

nace_sectors = [
    "agriculture", "mining and quarrying", "manufacturing",
    "electricity", "water supply and waste management",
    "construction", "wholesale and retail trade", "transportation and storage",
    "accommodation and food service", "information and communication",
    "financial and insurance", "real estate", "professional, scientific and technical",
    "administrative and support service", "public administration", 
    "education", "health and social work", "arts, entertainment and recreation"
    ]
print(f'Number of candidate sectors: {len(nace_sectors)}')

adapted_sectors = [
#    "agriculture and mining", 
    "manufacturing", "pharmaceutical and life sciences",
    "electricity and energy", 
#    "water supply and waste management",
    "construction", "wholesale and retail stores", 
    "transportation and storage", "accommodation, hotel, restaurants and food service", 
    "information and communication", "financial and insurance", 
    "real estate", "legal and consulting", "research and academia", 
    "public administration and international organizations", 
    "education, teaching and schools", 
    "medicine, health and hospitals", 
#    "social work", 
    "arts, entertainment and recreation", 
    "newspapers, television and media industry"
    ]
print(f'Number of candidate sectors: {len(adapted_sectors)}')
adapted_sectors = [ s + " activities" for s in adapted_sectors]

# sectors = [
#     "research and academia", "schools and teaching", "financial and insurance industry", 
#     "manufacturing industry", "pharmaceutical and life sciences industry", 
#     "politics and public administration", "hotels and restaurant industry", 
#     "retail, wholesale and stores industry", "electricity and energy industry",
#     "medicine and hospitals industry", "newspapers, television and media industry", 
#     "information, communication and telecommunication industry"
#     ]
res = ipl.posting_to_industry(
    text = text, classifier = classifier, sectors = adapted_sectors, 
    company_name = company
    # , retrieve_probas=True
    #, select_sentence_by="company_name"
    )
res["company"], res["industry"]

# check the selected sentences
import nltk
company_description = ipl.retrieve_company_description(nltk.sent_tokenize(text), classifier=classifier)
classifier(company_description, candidate_labels = adapted_sectors)


# => experiment with few-shot-classifiers instead... use big companies/organizations where we know the ground truth and label them for training.

# => experiment with question-answering pipline
question_answerer = pipeline("question-answering")
postings_lan["eng"]["company_name"][180:200]
n = 180
company = postings_lan["eng"]["company_name"][n]
text = postings_lan["eng"]["job_description"][n]
text = ipl.retrieve_company_description(sentences = nltk.sent_tokenize(text), classifier = classifier)
question_answerer(
    question="To which industry does the employer of this job position belong to?",
    context=". ".join(text),
)
text[1400:1420]
# => impressive in detection of the important parts but not enough context to deliver standardized results