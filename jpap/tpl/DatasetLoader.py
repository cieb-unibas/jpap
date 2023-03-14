import string

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class DatasetLoader():
    def __init__(self, source: str = "huggingface", dataset_id: str = "papluca/language-identification", target: str = "en", seed: int = 10032023):
        self.source = source
        self.dataset_id = dataset_id
        self.dataset = self.get_data()
        self.target = target
        self.seed = seed
    
    def get_data(self):
        if self.source == "huggingface":
            dat = load_dataset(path = self.dataset_id)
        else:
            dat = pd.read_csv(filepath_or_buffer = self.dataset_id)
        return dat
    
    def relabel(self, data):
        if self.source != "huggingface":
            assert "labels" in self.dataset.columns
        labelled = [l if l == self.target else "rest" for l in data["labels"]]
        return labelled

    def update(self):
        if self.source == "huggingface":
            df = {}
            for partition in ["train", "test", "validation"]:
                dat = self.dataset[partition].to_pandas()
                dat["labels"] = self.relabel(data = dat)
                df[partition] = dat
            self.dataset = df
        else:
            self.dataset["labels"] = self.relabel(data = self.dataset)

    def split(self, test_size):
        assert isinstance(self.dataset, pd.DataFrame), "Convert dataset to a pandas.DataFrame object."
        if isinstance(test_size, float):
            n_test = (len(self.dataset) * test_size)
            val_size =  n_test / (len(self.dataset) - n_test)
        else:
            val_size = test_size
        self.dataset = self.dataset.sample(fraction = 1, seed = self.seed) # shuffle
        df = {}
        df["train"], df["test"] = train_test_split(self.dataset, test_size=test_size, random_state=self.seed, stratify=True)
        df["train"], df["validation"] = train_test_split(df["train"], test_size=val_size, random_state=self.seed, stratify=True)
        self.dataset = df

    def label(self, partition, ouput_mode = "np"):
        self.label_dict = {"rest": 0, self.target: 1}
        out = np.array([1 if x == self.target else 0 for x in self.dataset[partition]["labels"]])
        if ouput_mode == "pt":
            return torch.from_numpy(out)
        else:
            return out
        
    def split_to_tokens(self, text):
        cleaned_text = "".join([c for c in text if c not in string.punctuation and c not in string.digits])
        tokenized_text = cleaned_text.lower().split()
        return tokenized_text

    def vocab(self, max_tokens: int = 50000, partition: str = "train"):
        counter = {}
        for text in self.dataset[partition]["text"]:
            tokens = self.split_to_tokens(text)
            for token in tokens:
                if token not in counter:
                    counter[token] = 0
                counter[token] += 1
        sorted_tokens = sorted(counter.items(), key = lambda x: x[1], reverse=True)
        sorted_tokens = [c[0] for c in sorted_tokens]
        if max_tokens:
            sorted_tokens = sorted_tokens[:max_tokens-2]
        sorted_tokens = ["<pad>", "<unk>"] + sorted_tokens
        self.vocabulary = {token: i for i, token in enumerate(sorted_tokens)}
        print("Vocabulary specified based on %s partition" % partition)
        return self

    def tokenize(self, text):
        processed_text = [self.vocabulary[t] if t in self.vocabulary.keys() else 1 for t in self.split_to_tokens(text)]
        return processed_text

    def tokenize_sequence(self, partition: str, output_mode: str, max_len: int = None):
        tokenized_sequence = [self.tokenize(text) for text in self.dataset[partition]["text"]]
        if max_len == None:
            max_len = max([len(x) for x in tokenized_sequence])
        tokenized_sequence = np.array([np.array(x + ([0] * (max_len - len(x))), dtype=np.int32) for x in tokenized_sequence], dtype=np.int32)
        if output_mode == "pt":
            return torch.from_numpy(tokenized_sequence)
        else:
            return tokenized_sequence

class LangDetectDataset(Dataset):
    def __init__(self, labels, texts) -> None:
        super().__init__()
        self.labels = labels
        self.texts = texts

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label
