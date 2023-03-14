import string

import numpy as np
from jpap.tpl import DatasetLoader
import torch

#### load and label the data:
data = DatasetLoader(source = "huggingface", dataset_id = "papluca/language-identification", target= "en")
y_train, y_val, y_test = data.label(partition="train", ouput_mode="pt"), data.label(partition="validation", ouput_mode="pt"), data.label(partition="test", ouput_mode="pt")

#### tokenize the texts and build a vocabulary:
def tokenizer(text: str) -> list:
    cleaned_text = "".join([c for c in text if c not in string.punctuation and c not in string.digits])
    tokenized_text = cleaned_text.lower().split()
    return tokenized_text

def vocab(texts, max_tokens: int = None) -> dict:
    counter = {}
    for text in texts:
        tokens = tokenizer(text)
        for token in tokens:
            if token not in counter:
                counter[token] = 0
            counter[token] += 1
    sorted_tokens = sorted(counter.items(), key = lambda x: x[1], reverse=True)
    sorted_tokens = [c[0] for c in sorted_tokens]
    if max_tokens:
        sorted_tokens = sorted_tokens[:max_tokens-2]
    sorted_tokens = ["<pad>", "<unk>"] + sorted_tokens
    vocabulary = {token: i for i, token in enumerate(sorted_tokens)}
    return vocabulary

def text_processing(text, vocabulary):
    processed_text = [vocabulary[t] if t in vocabulary.keys() else 1 for t in tokenizer(text)]
    return processed_text

def tokenize_text(partition: str, vocabulary: dict, output_mode: str, max_len: int = None):
    tokenized_sequence = [text_processing(text, vocabulary = vocabulary) for text in data.dataset[partition]["text"]]
    if max_len == None:
        max_len = max([len(x) for x in tokenized_sequence])
    tokenized_sequence = np.array([x + ([0] * (max_len - len(x))) for x in tokenized_sequence])
    if output_mode == "pt":
        return torch.from_numpy(tokenized_sequence)
    else:
        return tokenized_sequence


v = vocab(texts = data.dataset["train"]["text"], max_tokens=50000)

# include this in the dataset loader class
from torch.utils.data import Dataset
df_train = Dataset(labels = data.label(partition="train", ouput_mode="pt"), texts=tokenize_text(partition="train", vocabulary= v, output_mode="pt"))



#### tokenize the text:
# raw tokenization:
from torch.utils.data import DataLoader

class LanguageDetectDataset(Dataset):
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

df_train = LanguageDetectDataset(labels = data.dataset["train"]["labels"], texts=data.dataset["train"]["text"])
train_dataloader = DataLoader(df_train,batch_size=64, shuffle=True)

df_val = LanguageDetectDataset(labels = data.dataset["validation"]["labels"], texts=data.dataset["validation"]["text"])
val_dataloader = DataLoader(df_val,batch_size=64, shuffle=True)

df_test = LanguageDetectDataset(labels = data.dataset["test"]["labels"], texts=data.dataset["test"]["text"])
test_dataloader = DataLoader(df_test,batch_size=64, shuffle=True)
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

x_batch, y_batch = next(iter(train_dataloader))
len(x_batch)




# b) using xlm-roberta language specific tokens ------------
# https://colab.research.google.com/drive/15LJTckS6gU3RQOmjLqxVNBmbsBdnUEvl?usp=sharing#scrollTo=V_gbHRmNHEWU
# https://huggingface.co/xlm-roberta-base
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
tokenizer(data.dataset["train"]["text"][1], return_tensors = "pt")


