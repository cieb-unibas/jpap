#### könnte daraus auch eine class `DescExtractor` machen.
# attributes wären dann die postings, die company names.
# methods wären die einzelnen funktionen, welche an diesen dingen operieren.
class DescExtractor(object):
    def __init__(self, postings : list):
        self.postings = postings
        self.employer_desc = self.init_desc()
        self.tokenized_postings = self.tokenize()
    
    def init_desc(self):
        desc = [[""] for p in range(len(self.postings))]
        return desc

    def tokenize(self):
        for p in self.postings:
            self.tokenized_postings += nltk.sent_tokenize(p)
    
    def update_input(self):
        """
        Idee: nimm self.postings und wirf all jene sätze raus, die schon in `employer_desc` sind
        """
        updated_postings = []
        for p, d in zip(self.postings, self.employer_desc):
            text = np.array(nltk.sent_tokenize(p))
            desc = np.array(nltk.sent_tokenize(d))
            updated_text = text[~np.in1d(text, desc)]
            updated_text = " ".join(updated_text)
            updated_postings += updated_text
        return updated_postings

    def extract_by_name(self, employer_names):
        """
        """
        self.employer_names = employer_names
        text_inputs = self.update_postings()
        for input, company in zip(text_inputs, employer_names):
            sentences = nltk.sent_tokenize(text_inputs)

        # self.employer_names = employer_names
        # self.employer_desc =

    def extract_by_zsc(self, classifier, targets):
        """
        """