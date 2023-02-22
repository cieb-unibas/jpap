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
    
    def update(self):
        """
        """
        if not self.retrieved_by:
            return None
        else:
            updated_postings = []
            for p, d in zip(self.tokenized_input, self.employer_desc):
                tokenized_empdesc = nltk.sent_tokenize(d)
                updated_posting = [s for s in p if s not in tokenized_empdesc]
                updated_posting = " ".join(updated_posting)
                updated_postings.append(updated_posting)
            # update the attributes:
            self.input = updated_postings
            self.tokenized_input = self.tokenize_all()

    def by_name(self, employer_names):
        """
        """
        assert len(self.postings) == len(employer_names), "`employer_names` must have the same length as the number of postings supplied to `DescEctractor()`."
        
        self.employer_names = employer_names
        self.update()
        
        for i, t in enumerate(self.tokenized_input):
            # tokenize and retrieve relevant sentences
            company_sentences = [s for s in t if employer_names[i] in s.lower()]
            if len(company_sentences) > 1:
                company_description = [" ".join(company_sentences)]
            else:
                company_description = company_sentences

            # check if description is already available and will be enriched or not.
            if len(self.employer_desc) <= i:
                if company_description:
                    self.employer_desc += company_description
                else:
                    self.employer_desc += [""]
            else:
                if company_description:
                    self.employer_desc[i] += company_description[0]

        self.retrieved_by += ["name"]
        if len(self.retrieved_by) > 1:
            self.retrieved_by = list(set(self.retrieved_by))
    
    def by_zsc(self, classifier, targets):
        """
        """
        self.update()
        labels = targets + ["other"]

        for i, t in enumerate(self.tokenized_input):
            company_sentences = [classifier(s, candidate_labels=labels) for s in t]
            company_sentences = [s["sequence"] for s in company_sentences if s["labels"][0] in targets]

            if len(company_sentences) > 1:
                company_description = [" ".join(company_sentences)]
            else:
                company_description = company_sentences

            # check if description is already available and will be enriched or not.
            if len(self.employer_desc) <= i:
                if company_description:
                    self.employer_desc += company_description
                else:
                    self.employer_desc += [""]
            else:
                if company_description:
                    self.employer_desc[i] += company_description[0]

        self.retrieved_by += ["zsc"]
        if len(self.retrieved_by) > 1:
            self.retrieved_by = list(set(self.retrieved_by))
