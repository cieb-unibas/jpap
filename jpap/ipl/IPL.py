import os
import json

import torch
from transformers import AutoTokenizer

from .DescExtractor import DescExtractor

class IPL(object):
    """
    Pipeline for predicting industry labels based on different classifiers.
    """
    def __init__(self, classifier : str = "pharma"):
        self.classifier_type = classifier
        self.path = "/scicore/home/weder/GROUP/Innovation/05_job_adds_data/augmentation_data/ipl_classifer_" + self.classifier_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = self.load_classifier()
        self.labels = self.load_labels()
        self.tokenizer = self.load_tokenizer()

    def load_classifier(self):
        file = self.path + ".pt"
        classifier = torch.load(file, map_location=torch.device(self.device))
        return classifier

    def load_labels(self):
        file = self.path + ".json"
        with open(file, "r", encoding = "UTF-8") as f:
            labels = json.load(f)
        return labels
    
    def load_tokenizer(self):
        tokenizer_path = "/scicore/home/weder/GROUP/Innovation/05_job_adds_data/hf_models/xlm-roberta-base"
        if os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        return tokenizer
    
    def __call__(self, postings, company_names: list = None, return_probas = False):
        # get the employer descriptions
        extractor = DescExtractor(postings=postings)
        if company_names is not None:
            employer_description = extractor(employer_names=company_names, use_zsc=True)
        else:
            employer_description = extractor(use_zsc=True)
        # take care of NAs
        employer_description = [text if text else "no relevant information" for text in employer_description]
        # tokenize the employer descriptions
        x_tokenized = self.tokenizer(employer_description, return_tensors="pt", truncation=True, max_length=128, padding=True)
        # predict industry labels
        self.classifier.to(self.device)
        self.classifier.eval()
        with torch.no_grad():
            x = x_tokenized["input_ids"].to(self.device)
            mask = x_tokenized["attention_mask"].to(self.device)
            outputs = self.classifier(x, attention_mask = mask)
            predicted_probas = outputs["logits"]
            predicted_classes = torch.argmax(predicted_probas, dim = 1).tolist()
        # convert to original labels
        predicted_classes = [self.labels[str(k)] for k in predicted_classes]
        
        if return_probas:
            return predicted_classes, predicted_probas
        else:
            return predicted_classes