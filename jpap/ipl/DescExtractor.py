from transformers import pipeline
import nltk

class DescExtractor(object):
    def __init__(self, postings):
        self.postings = postings
        self.employer_desc = []
        self.retrieved_by = []
        self._input = postings
        self._tokenized_input = self.tokenize_all()
    
    def tokenize_all(self):
        """
        Tokenize all postings text by splitting them up into sentences.
        """
        tokenized_input = [nltk.sent_tokenize(p) for p in self._input]
        return tokenized_input 
    
    def _update(self, tokenized_posting, employer_description):
        """
        Update a postings text to determine what part of the original text is a candidate for feature extraction.
        """
        updated_posting = [s for s in tokenized_posting if s not in nltk.sent_tokenize(employer_description)]
        updated_posting = " ".join(updated_posting)
        return updated_posting

    def _update_all(self):
        """
        Update all text inputs to determine what part of the original text is a candidate for feature extraction.
        """
        if not self.retrieved_by:
            return None
        else:
            updated_postings = []
            for p, d in zip(self._tokenized_input, self.employer_desc):
                updated_posting = self._update(p, d)
                updated_postings.append(updated_posting)
            self._input = updated_postings
            self._tokenized_input = self.tokenize_all()

    def _assign_empdesc(self, desc: str, idx: int) -> None:
        """
        Stores retrieved employer descriptions.
        """
        if len(self.employer_desc) > idx:
            if len(self.employer_desc[idx]) > 0:
                self.employer_desc[idx] += " " + desc
            else:
                self.employer_desc[idx] += desc
        else:
            self.employer_desc.append(desc)

    def _log_retrieved(self, by: str) -> None:
        """
        Logger indicating by which methods descriptions have been extracted.
        """
        self.retrieved_by += [by]
        if len(self.retrieved_by) > 1:
            self.retrieved_by = list(set(self.retrieved_by))
    
    def sentences_by_name(self, employer_names, log_number = 200, silent_mode = True):
        """
        Extract all sentences from job postings that feature the company name.
        """
        assert len(self.postings) == len(employer_names), "`employer_names` must have the same length as the number of postings supplied to `DescEctractor()`."
        self.employer_names = employer_names
        self._update_all()
        for i, t in enumerate(self._tokenized_input):
            employer_desc = " ".join([s for s in t if employer_names[i] in s.lower()])
            self._assign_empdesc(desc = employer_desc, idx = i)
            if i % log_number == 0 and not silent_mode:
                print("Applied name-based extraction to %d postings" % i)
        self._log_retrieved(by = "name")
   
    def sentences_by_zsc(
            self, classifier, targets: list[str], 
            excluding_classes: list[str] = ["address", "benefits"], 
            log_number: int = 200, silent_mode = True
            ):
        """
        Extract all sentences from job postings that are labelled by a zero-shot-classifier to
        one or more target classes `targets`.
        """
        self._update_all()
        labels = targets + ["other"]
        if excluding_classes:
            labels += excluding_classes
        for i, t in enumerate(self._tokenized_input):
            employer_sentences = [classifier(s, candidate_labels = labels) for s in t]
            employer_desc = " ".join([s["sequence"] for s in employer_sentences if s["labels"][0] in targets])
            self._assign_empdesc(desc = employer_desc, idx = i)
            if i % log_number == 0 and not silent_mode:
                print("Applied zero-shot extration to %d postings" % i)
        self._log_retrieved(by = "zsc")

    def __call__(self, employer_names: list = None, use_zsc = True):
        if employer_names is not None:
            self.sentences_by_name(employer_names=employer_names)
        if use_zsc:
            self.zsc_path = "/scicore/home/weder/GROUP/Innovation/05_job_adds_data/hf_models/multilingual-MiniLMv2-L6-mnli-xnli/"
            classifier = pipeline(task = "zero-shot-classification", model = self.zsc_path)
            self.zsc_targets = ["who we are", "who this is", "industry or sector"]
            self.sentences_by_zsc(classifier=classifier, targets = self.zsc_targets)
        employer_descriptions =  [None if x == "" else x for x in self.employer_desc]
        return employer_descriptions


