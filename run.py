from pipeline.preprocessing import DataRetriever
import torch
from transformers import pipeline

#### connect to JPOD
retriever = DataRetriever(db_path="C:/Users/matth/Desktop/jpod_test.db")

#### get postings and language information:
postings = retriever.get_postings()
postings_lan = {}
for lan in ["eng", "ger", "fre", "ita"]:
    postings_lan[lan] = postings[postings["text_language"] == lan]
    print("%d postings stored for language '%s'" % (len(postings_lan[lan]), lan))

# ideas:
# 1) split into sentences and label sentences via zero-shot-learning that describe a company
# 2) split into paragraphs and label these via zero-shot-learning
# 3) then retrieve these paragraphs and translate them.


#### translate postings to 'eng':
posting = list(postings_lan["eng"]["job_description"])[16]
posting = "Credit Suisse is a leading global wealth manager with strong investment banking capabilities. Headquartered in Zurich, Switzerland, we have a global reach with operations in about 50 countries and employ more than 45,000 people from over 150 different nations."
classifier = pipeline("zero-shot-classification")
classifier(posting, candidate_labels = ["research and education sector", "banking and insurance industry", "machinery industry", "pharmaceutical industry", "other industry"])    


