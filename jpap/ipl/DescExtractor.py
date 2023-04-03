import nltk

from jpap.tpl.detect_language import detect_language

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

    def _posting_language(self, model_id_or_path: str = "juliensimon/xlm-v-base-language-id", n_tokens: int = 20):
        texts = [" ".join(i[0].split(" ")[:n_tokens]).lower() for i in self.tokenized_input]
        languages = detect_language(text = texts, model_id_or_path = model_id_or_path)
        self.languages = languages
        print("Posting languages labelled.")

    def _multilang_label_dict(self, labels, translator, source_language = "en"):
        """"
        Translate chosen labels of a certain language to all languages present in the dataset using a translator.
                ---> Test if this is even necessary...
        """
        d = {}
        for lan in list(set(self.languages)):
            d[lan] = [translator.translate(text = label, dest = lan, src = source_language).text for label in labels]
        return d
    
    def by_zsc(self, classifier, targets, target_translator):
        """
        """
        self.update_all()
        labels = targets + ["other"]
        label_dict = self._multilang_label_dict(labels = labels, source_language="en", translator=target_translator)
        for i, t in enumerate(self.tokenized_input):
            employer_sentences = [classifier(s, candidate_labels = label_dict[self.language[i]]) for s in t]
            employer_desc = " ".join([s["sequence"] for s in employer_sentences if s["labels"][0] in targets])
            self.assign_empdesc(desc = employer_desc, idx = i)
        self.log_retrieved(by = "zsc")

texts[22]