import nltk
## => mach dies als class...
## attributes können dann tokenized_text, company_description, probas, industry etc. sein...

# class SectorClassifier():
#     def __init__(self, classifier, sectors):
#         self.classifier = classifier,
#         self.sectors = sectors

def retrieve_companyname_sentences(sentences, company_name, verbose = False):
    company_sentences = [s for s in sentences if company_name in s.lower()]
    if verbose:
        print("Classified %d sentences as relevant for company description" % len(company_sentences))
    return company_sentences

def retrieve_company_description(sentences, classifier, verbose = False):
    candidate_labels = ["who we are", "other"]
    company_sentences = [classifier(s, candidate_labels=candidate_labels) for s in sentences]
    company_sentences = [s["sequence"] for s in company_sentences if s["labels"][0] == candidate_labels[0]]
    if verbose:
        print("Classified %d sentences as relevant for company description using zero-shot-learning" % len(company_sentences))
    return company_sentences

def posting_to_industry(text, classifier, sectors, select_sentence_by = "zsc", company_name = None, retrieve_probas = False):
    """
    Assign a job posting to a set of industries
    """
    assert select_sentence_by in ["zsc", "company_name", None], "`select_sentence_by` must be one of `zsc`, `company_name`, or `None`"
    
    posting_sentences = nltk.sent_tokenize(text)
    
    # define the (sub-)sample of text that is provided to the classifi3er
    if select_sentence_by == "zsc":
        company_description = " ".join(retrieve_company_description(sentences = posting_sentences, classifier = classifier))
    elif select_sentence_by == "company_name":
        assert company_name, "A `company_name` has to be provided if `select_sentence_by` is set to 'company_name'."
        company_description = " ".join(retrieve_companyname_sentences(sentences = posting_sentences, company_name = company_name))
        assert len(company_description) > 0, "No sentence mentioning the company's name could be found. Please re-try using zsc."
    else:
        company_description = text
    
    # classify the posting
    preds = classifier(company_description, candidate_labels = sectors)
    industry = preds["labels"][0]
    preds = list(zip(preds["labels"], [round(s, 3) for s in preds["scores"]]))
    
    # output:
    if retrieve_probas:
        return [("company", company_name)] + preds
    else:
        return {"postings_text": text, "company": company_name, "industry": industry}