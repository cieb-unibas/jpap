from pipeline.preprocessing import DataRetriever
from bs4 import BeautifulSoup
import nltk
from transformers import pipeline


#### connect to JPOD and retrieve data
retriever = DataRetriever(db_path="C:/Users/matth/Desktop/jpod_test.db")

#### get postings and language information:
postings = retriever.get_postings(html = True, institution_name=True)
postings_lan = {}
for lan in ["eng", "ger", "fre", "ita"]:
    postings_lan[lan] = postings[postings["text_language"] == lan].reset_index().drop("index", axis=1)
    print("%d postings stored for language '%s'" % (len(postings_lan[lan]), lan))

#### split postings into paragraphs:
postings_lan["eng"]["company_name"][:40]
n = 7
company = postings_lan["eng"]["company_name"][n]
text = postings_lan["eng"]["job_description"][n]
html_text = postings_lan["eng"]["html_job_description"][n]

# via bs4
soup = BeautifulSoup(html_text)
paragraphs = soup.find_all("p")
paragraphs = [p.contents for p in paragraphs]
paragraphs = [p for p in paragraphs if len(p) > 0]
assert sum([len(p) for p in paragraphs]) == len(paragraphs)
paragraphs = [p[0].lower() for p in paragraphs if isinstance(p[0], str)]
# => html tags are too diverse and weird to establish meaningful rules...

# via nltk
posting_sent = nltk.tokenize.sent_tokenize(text)

# manual retrieval of company relevant statements via paragraphs
company_statement_man_para = [p for p in paragraphs if company in p]
print("Classified %d paragraphs as relevant for company description" % len(company_statement_man_para))

# manual retrieval of company relevant statements via sentences
company_statement_man_sen = [s for s in posting_sent if company in s.lower()]
print("Classified %d sentences as relevant for company description" % len(company_statement_man_sen))
posting = " ".join(company_statement_man_sen)

# zero-shot classification paragraphs:
classifier = pipeline("zero-shot-classification")
candidate_labels = ["company description", "other"]

res = [classifier(p, candidate_labels=candidate_labels) for p in paragraphs]
company_statement_zsc_para = [p["sequence"] for p in res if p["labels"][0] == candidate_labels[0]]
print("Classified %d paragraphs as relevant for company description using zero-shot-learning" % len(company_statement_zsc_para))

res = [classifier(s, candidate_labels=candidate_labels) for s in posting_sent]
company_statement_zsc_sent = [s["sequence"] for s in res if s["labels"][0] == candidate_labels[0]]
print("Classified %d sentences as relevant for company description using zero-shot-learning" % len(company_statement_zsc_sent))
posting = " ".join(company_statement_zsc_sent)

# ideas:
# 1) split into sentences and label sentences via zero-shot-learning that describe a company
# 2) split into paragraphs and label these via zero-shot-learning
# 3) then retrieve these paragraphs and translate them.


#### translate postings to 'eng':
posting = list(postings_lan["eng"]["job_description"])[16]
posting = "Credit Suisse is a leading global wealth manager with strong investment banking capabilities. Headquartered in Zurich, Switzerland, we have a global reach with operations in about 50 countries and employ more than 45,000 people from over 150 different nations."
classifier = pipeline("zero-shot-classification")
classifier(posting, candidate_labels = ["research and education sector", "banking and insurance industry", "machinery industry", "pharmaceutical industry"])    


