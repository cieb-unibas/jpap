import nltk

class DescExtractor(object):
    def __init__(self, postings):
        self.postings = postings
        self.employer_desc = []
        self.retrieved_by = []
        self.input = postings
        self.tokenized_input = self.tokenize_all()
    
    def tokenize_all(self):
        tokenized_input = [nltk.sent_tokenize(p) for p in self.input]
        return tokenized_input 
    
    def update(self, tokenized_posting, employer_description):
        updated_posting = [s for s in tokenized_posting if s not in nltk.sent_tokenize(employer_description)]
        updated_posting = " ".join(updated_posting)
        return updated_posting

    def update_all(self):
        """
        """
        if not self.retrieved_by:
            return None
        else:
            updated_postings = []
            for p, d in zip(self.tokenized_input, self.employer_desc):
                updated_posting = self.update(p, d)
                updated_postings.append(updated_posting)
            self.tokenized_input = self.tokenize_all()

    def assign_empdesc(self, desc: str, idx: int) -> None:
        if len(self.employer_desc) > idx:
            if len(self.employer_desc[idx]) > 0:
                self.employer_desc[idx] += " " + desc
            else:
                self.employer_desc[idx] += desc
        else:
            self.employer_desc.append(desc)

    def log_retrieved(self, by: str) -> None:
        self.retrieved_by += [by]
        if len(self.retrieved_by) > 1:
            self.retrieved_by = list(set(self.retrieved_by))
    
    def by_name(self, employer_names):
        """
        """
        assert len(self.postings) == len(employer_names), "`employer_names` must have the same length as the number of postings supplied to `DescEctractor()`."
        self.employer_names = employer_names
        self.update_all()
        for i, t in enumerate(self.tokenized_input):
            employer_desc = " ".join([s for s in t if employer_names[i] in s.lower()])
            self.assign_empdesc(desc = employer_desc, idx = i)
        self.log_retrieved(by = "name")
    
    def by_zsc(self, classifier, targets):
        """
        """
        self.update_all()
        labels = targets + ["other"]
        for i, t in enumerate(self.tokenized_input):
            employer_sentences = [classifier(s, candidate_labels=labels) for s in t]
            employer_desc = " ".join([s["sequence"] for s in employer_sentences if s["labels"][0] in targets])
            self.assign_empdesc(desc = employer_desc, idx = i)
        self.log_retrieved(by = "zsc")