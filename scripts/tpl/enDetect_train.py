#### train the language detection model necessary for the translation pipeline (tpl)
import numpy as np
from jpap.tpl import DatasetLoader

#### load and label the data:
data = DatasetLoader(source = "huggingface", dataset_id = "papluca/language-identification", target= "en")
y_train, y_val, y_test = data.label(partition="train"), data.label(partition="validation"), data.label(partition="test")

#### build a vocabulary and tokenize the texts:



#### tokenize the text:
# raw tokenization:
from torch.utils.data import Dataset
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


