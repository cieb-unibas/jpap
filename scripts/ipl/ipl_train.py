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

def set_paths():
    try:
        HOME = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
    except:
        HOME = os.getcwd()
    sys.path.append(HOME)
    return HOME

def load_labelled(n_postings : int = None, home_dir: str = set_paths()) -> pd.DataFrame:
    """
    Load n postings' employer descriptions that were manually labelled to industries.
    """
    df = pd.read_csv(home_dir + "/data/created/industry_train.csv")
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

def finetune(model, n_epochs : int, train_loader, eval_loader, device : str):
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
            # clean up gpu memory
            del x
            del y
            del mask
            del outputs
            torch.cuda.empty_cache()
            # update weights
            optim.zero_grad()
            loss.backward()
            optim.step()
            del loss
            # logs
            duration = time.time() - batch_start_time
            if i % 5 == 0:
                print("%d/%d data batches processed | Training time batch: %d seconds" % (i+1, len(train_loader), duration))

        # evaluate
        duration = datetime.timedelta(seconds = time.time() - epoch_start_time).total_seconds() / 60
        train_acc = float(correct_samples) / n_samples
        val_acc = evaluate_finetuned(model = model, eval_loader = eval_loader, device = device)
        print("Epoch %d/%d | Training Accuracy: %.3f | Validation Accuracy: %.3f | Training time epoch: %.3f minutes" % (epoch + 1, n_epochs, train_acc, val_acc, duration))

    print("Training finished. Returning fine-tuned model")
    return model

# memory run-out
# https://www.kaggle.com/getting-started/140636
# https://stackoverflow.com/questions/63145729/how-to-make-sure-pytorch-has-deallocated-gpu-memory
# torch.backends.cudnn.deterministic = True
# torch.cuda.empty_cache()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# => another option:
# import gc
# gc.collect()
# torch.cuda.empty_cache()

# could apparently also be an incompatible cuda-pytorch issue
# https://stackoverflow.com/questions/54374935/how-to-fix-this-strange-error-runtimeerror-cuda-error-out-of-memory

if __name__ == "__main__":
    
    # load data & split data
    df = load_labelled()
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x = df["employer_description"], y = df["industry"], 
                                                            validation_set = True,  test_size=0.15, val_size=0.25,
                                                            random_state = 27042023)
    print("Number of training samples: %d" %len(x_train))
    print("Number of validation samples: %d" %len(x_val))
    print("Number of testing samples: %d" %len(x_test))

    # load encoder, tokenizer and model
    le = get_label_encoder(y = df["industry"])
    tokenizer, model = load_xlm_pretrained(n_targets= len(le.classes_))
    print("Label encoder, pre-trained tokenizer and pre-trained model checkpoint loaded.")

    # tokenize the data
    le = get_label_encoder(y = df["industry"])
    x_train, x_val, x_test = text_tokenizer(text_partitions = [x_train, x_val, x_test])
    y_train, y_val, y_test = encode_labels(label_encoder=le, y = y_train)\
        , encode_labels(label_encoder=le, y = y_val)\
        , encode_labels(label_encoder=le, y = y_test) 

    # data pipelines
    BATCH_SIZE = 16
    train_dl = DataLoader(
        IndustryClassificationDataset(targets = y_train, texts = x_train),
        batch_size = BATCH_SIZE, shuffle = True
        )
    val_dl = DataLoader(
        IndustryClassificationDataset(targets = y_val, texts = x_val),
        batch_size = BATCH_SIZE, shuffle = False
        )

    # configurations for finetuning the model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 2
    model.to(DEVICE)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)
    print("Training model with %d batches per epoch and %d epochs" % (len(train_dl), EPOCHS))
    finetuned_model = finetune(
        model = model, n_epochs = EPOCHS, 
        train_loader = train_dl, eval_loader = val_dl, 
        device= DEVICE
        )

    # if final: retrain on everything and save