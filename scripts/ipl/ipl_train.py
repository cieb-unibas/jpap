import random
import os
import sys
import time
import datetime

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_labelled(path: str, n_postings : int = None) -> pd.DataFrame:
    """
    Load n postings' employer descriptions that were manually labelled to industries.
    """
    df = pd.read_csv(path)
    if n_postings:
        random.seed(1)
        out_postings = random.choices(range(len(df)), k = n_postings)
        df = df.iloc[out_postings, :]
        df = df.reset_index(drop=True)
    return df

def load_xlm_pretrained(path_to_model = None, n_targets: int = 16):
    if path_to_model:
        tokenizer = AutoTokenizer.from_pretrained(path_to_model)
        model = AutoModelForSequenceClassification.from_pretrained(path_to_model, num_labels = n_targets)
    else:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels = n_targets)
    return tokenizer, model

def split_data(x, y, validation_set = True, test_size: float = 0.15, val_size: float = 0.25, random_state = None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= test_size, random_state=random_state)
    if validation_set:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=random_state)
        return x_train, x_val, x_test, y_train, y_val, y_test
    else:
        return x_train, x_test, y_train, y_test

def get_label_encoder(y):
    le = LabelEncoder()
    le.fit(y)
    return le

def encode_labels(label_encoder, y, torch_dtype = torch.int32):
    labels = torch.tensor(label_encoder.transform(y), dtype=torch_dtype)
    return labels

def text_tokenizer(text_partitions: list, return_tensors="pt", 
    truncation=True, max_length=128, padding=True):
    for i, t in enumerate(text_partitions):
        text_partitions[i] = tokenizer(t.to_list(), return_tensors=return_tensors,
                                       truncation=truncation, max_length=max_length,
                                       padding = padding)
    
    return text_partitions[0], text_partitions[1], text_partitions[2] 


class IndustryClassificationDataset(Dataset):
    def __init__(self, targets, texts) -> None:
        super().__init__()
        self.targets = targets
        self.texts = texts

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.texts.items()}
        item["labels"] = self.targets[idx]
        return item

def evaluate_finetuned(model, eval_loader, device):
    """
    Evaluate on testing set
    """
    with torch.no_grad():
        model.to(device)
        correct_samples, n_samples = 0, 0
        for batch in eval_loader:
            # define inputs and labels:
            x = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].type(torch.LongTensor).to(device)
            # forward pass:
            outputs = model(x, attention_mask = mask, labels = y)
            n_samples += y.size(0)
            correct_samples += (torch.argmax(outputs["logits"], dim = 1) == y).sum().item()
    acc = float(correct_samples) / n_samples
    return acc

def finetune(model, n_epochs : int, train_loader, eval_loader, device : str, return_finetuned_model = True):
    """
    Finetune a certain model using a training and evaluation pipeline.
    """
    
    model.to(device)
    
    for epoch in range(n_epochs):
        
        epoch_start_time = time.time()
        model.train()
        correct_samples, n_samples = 0, 0

        for i, batch in enumerate(train_loader):
            batch_start_time = time.time()
            # define inputs and labels:
            x = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].type(torch.LongTensor).to(device)
            # forward pass:
            outputs = model(x, attention_mask = mask, labels = y)
            loss = outputs["loss"]
            n_samples += y.size(0)
            correct_samples += (torch.argmax(outputs["logits"], dim = 1) == y).sum().item()
            # gradient calculation and update weights
            optim.zero_grad()
            loss.backward()
            optim.step()
            # logs
            duration = time.time() - batch_start_time
            if i % 5 == 0:
                print("Epooch %d: %d/%d data batches processed | Training time batch: %d seconds" % (epoch+1, i+1, len(train_loader), duration))

        # evaluate
        duration = datetime.timedelta(seconds = time.time() - epoch_start_time).total_seconds() / 60
        train_acc = float(correct_samples) / n_samples
        val_acc = evaluate_finetuned(model = model, eval_loader = eval_loader, device = device)
        print("Epoch %d/%d | Training Accuracy: %.3f | Validation Accuracy: %.3f | Training time epoch: %.3f minutes" % (epoch + 1, n_epochs, train_acc, val_acc, duration))

    if return_finetuned_model:
        print("Training finished. Returning fine-tuned model")
        return model
    else:
        print("Training finished.")

if __name__ == "__main__":
    
    # training parameters:
    TRAIN_DAT = "/scicore/home/weder/GROUP/Innovation/05_job_adds_data/augmentation_data/industry_train.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    EPOCHS = 5
    print("Training the model on %s over %d epochs in batches of size %d." % (DEVICE, EPOCHS, BATCH_SIZE))
    
    # load data & split data
    df = load_labelled(path = TRAIN_DAT)
    print("Total number of samples in the dataset: ", len(df))
    print("Class distribution:")
    print(df.groupby(["industry"])["industry"].count().sort_values(ascending=False))
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x = df["employer_description"], y = df["industry"], 
                                                            validation_set = True,  test_size=0.15, val_size=0.25,
                                                            random_state = 27042023)
    print("Number of training samples: %d" %len(x_train))
    print("Number of validation samples: %d" %len(x_val))
    print("Number of testing samples: %d" %len(x_test))

    # load encoder, tokenizer and model
    le = get_label_encoder(y = df["industry"])
    tokenizer, model = load_xlm_pretrained(
        path_to_model = "../hf_models/xlm-roberta-base", 
        n_targets = len(le.classes_)
        )

    # tokenize the data
    x_train, x_val, x_test = text_tokenizer(text_partitions = [x_train, x_val, x_test])
    y_train, y_val, y_test = encode_labels(label_encoder=le, y = y_train), encode_labels(label_encoder=le, y = y_val), encode_labels(label_encoder=le, y = y_test) 

    # data pipelines
    train_dl = DataLoader(
        IndustryClassificationDataset(targets = y_train, texts = x_train),
        batch_size = BATCH_SIZE, shuffle = True
        )
    val_dl = DataLoader(
        IndustryClassificationDataset(targets = y_val, texts = x_val),
        batch_size = BATCH_SIZE, shuffle = False
        )

    # Baseline accuracies:
    zero_r = max(pd.Series(y_train).value_counts() / len(y_train))
    random_guessing = sum(pd.Series(y_train).value_counts() / len(y_train) ** 2)
    print("Baseline accuracies for the training dataset:\n ZeroR: %4.3f\n RandomGuessing: %4.3f" % (zero_r, random_guessing))

    # configurations for finetuning the model
    model.to(DEVICE)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)
    finetuned_model = finetune(
        model = model, n_epochs = EPOCHS, 
        train_loader = train_dl, eval_loader = val_dl, 
        device= DEVICE, return_finetuned_model=False
        )    

    # if final: retrain on everything and save
    # finetuned_model.save_pretrained(SAVE_PATH)