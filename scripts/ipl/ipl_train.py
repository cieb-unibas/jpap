import random
import time
import datetime
import sys
import os
import json

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

sys.path.append(os.getcwd())
from jpap.preprocessing import subsample_df

def relabel(df, label_column: str = "industry", 
            relabel_dict_path: dict = os.path.join(os.getcwd(), "data/raw/macro_industry_mapings.json")) -> pd.DataFrame:
    file = os.path.join(relabel_dict_path)
    with open(file, "r", encoding = "UTF-8") as f:
        label_dict = json.load(f)
    df[label_column] = df[label_column].replace(label_dict)
    return df

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

def blind_employer_names(df, description_column: str = "employer_description", name_column: str = "company_name", replacement: str = "blinded_name"):
    assert isinstance(df, pd.DataFrame), "`df`must be of type pandas.DataFrame"
    df[description_column] = df.apply(lambda x: x[description_column].replace(x[name_column], replacement), axis = 1)   
    return df

def load_xlm_pretrained(path_to_model = None, n_targets: int = 16):
    if path_to_model:
        tokenizer = AutoTokenizer.from_pretrained(path_to_model)
        model = AutoModelForSequenceClassification.from_pretrained(path_to_model, num_labels = n_targets)
    else:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels = n_targets)
    return tokenizer, model

def random_data_split(df, x: str = "employer_description", y: str = "industry", validation_set = True, 
                      test_size: float = 0.15, val_size: float = 0.25, random_state = 10082023):
    x_train, x_test, y_train, y_test = train_test_split(df[x], df[y], test_size= test_size, random_state=random_state, stratify=df[y])
    if validation_set:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)
        return x_train, x_val, x_test, y_train, y_val, y_test
    else:
        return x_train, x_test, y_train, y_test


def split_by_employers(df, x = "employer_description", y = "industry",
                       validation_set = True, test_size: float = 0.15, 
                       val_size: float = 0.25, random_state = 8082023
                       ):
    company_counts = pd.DataFrame(df.groupby(["company_name"])["company_name"].count().sample(frac = 1, random_state = random_state))
    company_counts["cumsum"] = company_counts["company_name"].cumsum()

    def get_partition_sample(df, target_column, employers):
        output = df.loc[df["company_name"].isin(employers), target_column].reset_index(drop=True)
        return output
    
    # define the test partition:
    n_test_postings = int(len(df) * test_size)
    test_companies = company_counts.loc[company_counts["cumsum"] < n_test_postings, ].index.tolist()
    x_test = get_partition_sample(df = df, target_column=x, employers=test_companies)
    y_test = get_partition_sample(df = df, target_column=y, employers=test_companies)

    # define training and validation partitions
    train_companies = company_counts.loc[company_counts["cumsum"] > n_test_postings, "company_name"]

    if validation_set:
        train_companies = pd.DataFrame(train_companies)
        train_companies["cumsum"] = train_companies["company_name"].cumsum()
        n_valid_postings = int(max(train_companies["cumsum"]) * val_size)
        val_companies = train_companies.loc[train_companies["cumsum"] < n_valid_postings, ].index.tolist()
        assert not any(c in val_companies for c in test_companies)

        x_val = get_partition_sample(df = df, target_column=x, employers=val_companies)
        y_val = get_partition_sample(df = df, target_column=y, employers=val_companies)
        
        train_companies = train_companies.loc[train_companies["cumsum"] > n_valid_postings, ].index.tolist()
        assert not any(c in train_companies for c in val_companies)
        x_train = get_partition_sample(df = df, target_column=x, employers=train_companies)
        y_train = get_partition_sample(df = df, target_column=y, employers=train_companies)

        return x_train, x_val, x_test, y_train, y_val, y_test
    else:
        train_companies = train_companies.index.tolist()
        assert not any(c in train_companies for c in test_companies)    
        x_train = get_partition_sample(df = df, target_column=x, employers=train_companies)
        y_train = get_partition_sample(df = df, target_column=y, employers=train_companies)

        return x_train, x_test, y_train, y_test

def partition_data(df, x, y, split_by : str = "random",
                   validation_set = True, test_size: float = 0.15,
                   val_size: float = 0.25, random_state = 8082023):
    if split_by == "random":
        return random_data_split(df = df, x = x, y = y, validation_set=validation_set, test_size=test_size, val_size=val_size, random_state=random_state)
    elif split_by == "employers":
        return split_by_employers(df = df, x = x, y = y, validation_set=validation_set, test_size=test_size, val_size=val_size, random_state=random_state)
    else:
        raise ValueError("Set the `split_by` parameter to either 'random' or 'employers'.")

def get_label_encoder(y):
    le = LabelEncoder()
    le.fit(y)
    return le

def encode_labels(label_encoder, y, torch_dtype = torch.int32):
    labels = torch.tensor(label_encoder.transform(y), dtype=torch_dtype)
    return labels

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

def evaluate_finetuned(model, eval_loader, device, return_acc = True, return_predicted_classes = False):
    """
    Evaluate on testing set
    """
    with torch.no_grad():
        model.to(device)
        predicted_classes = {"y_pred": [], "y_true": []}
        correct_samples, n_samples = 0, 0
        for batch in eval_loader:
            # define inputs and labels:
            x = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].type(torch.LongTensor).to(device)
            predicted_classes["y_true"] += batch["labels"].tolist()
            # forward pass:
            outputs = model(x, attention_mask = mask, labels = y)
            predicted_outputs = torch.argmax(outputs["logits"], dim = 1)
            predicted_classes["y_pred"] += predicted_outputs.tolist()
            n_samples += y.size(0)
            correct_samples += (predicted_outputs == y).sum().item()
    acc = float(correct_samples) / n_samples
    if return_acc and not return_predicted_classes:
        return acc
    elif return_predicted_classes and not return_acc:
        return predicted_classes
    else:
        return acc, predicted_classes

def finetune(model, n_epochs : int, train_loader, device : str, optimizer, 
             eval_loader = None, return_finetuned_model: bool = True, silent_training = False):
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # logs
            duration = time.time() - batch_start_time
            if i % 5 == 0 and not silent_training:
                log_message = """Epoch %d: %d/%d data batches processed | Training Accuracy: %.3f | Training time batch: %.3f seconds"""\
                    % (epoch+1, i+1, len(train_loader), float(correct_samples / n_samples), duration)
                print(log_message)

        # evaluate
        duration = datetime.timedelta(seconds = time.time() - epoch_start_time).total_seconds() / 60
        train_acc = float(correct_samples) / n_samples
        if eval_loader and not silent_training:
            val_acc = evaluate_finetuned(model = model, eval_loader = eval_loader, device = device)
            print("------Epoch %d/%d | Training Accuracy: %.3f | Validation Accuracy: %.3f | Training time epoch: %.3f minutes------"\
                % (epoch + 1, n_epochs, train_acc, val_acc, duration))

    if return_finetuned_model:
        print("Training finished. Returning fine-tuned model")
        return model
    else:
        print("Training finished.")

def get_model_report(model, eval_loader, label_encoder, device = "cuda"):
    predicted_classes = evaluate_finetuned(
        model = model, eval_loader = eval_loader, 
        device = device, return_acc=False, return_predicted_classes=True)
    report = classification_report(
        y_true=predicted_classes["y_true"], y_pred=predicted_classes["y_pred"], 
        labels = list(range(len(label_encoder.classes_))),
        digits=3, target_names=label_encoder.classes_, zero_division=1)
    return report

if __name__ == "__main__":
    
    # data parameters:
    TRAIN_DAT = "/scicore/home/weder/GROUP/Innovation/05_job_adds_data/augmentation_data/industry_train.csv"
    ONLY_UNIQUE_EMPLOYERS = False
    MACRO_INDUSTRY_GROUPS = True
    MAX_SAMPLES_PER_INDUSTRY = 400

    # training data parameters:
    BLIND_EMPLOYER_NAMES = True
    SPLIT_BY = "employers"

    # training process parameters:
    SAVE_MODEL_PATH = "/scicore/home/weder/GROUP/Innovation/05_job_adds_data/augmentation_data/ipl_classifer.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    EPOCHS = 2
    print("Training the model on %s over %d epochs in batches of size %d." % (DEVICE, EPOCHS, BATCH_SIZE))
    
    # load data
    if ONLY_UNIQUE_EMPLOYERS:
        df = load_labelled(path=TRAIN_DAT).drop_duplicates(subset=["company_name"]).reset_index(drop=True)
    else:
        df = load_labelled(path = TRAIN_DAT)
    if MACRO_INDUSTRY_GROUPS:
        df = relabel(df=df, label_column="industry")
    if MAX_SAMPLES_PER_INDUSTRY:
        df = subsample_df(df=df, group_col="industry", max_n_per_group = MAX_SAMPLES_PER_INDUSTRY)
    print("Total number of samples in the dataset: ", len(df))
    print("Class distribution:")
    print(df.groupby(["industry"])["industry"].count().sort_values(ascending=False))

    # process data and split into partitions
    if BLIND_EMPLOYER_NAMES:
        df = blind_employer_names(df=df, description_column="employer_description", name_column="company_name", replacement="this company")
    x_train, x_test, y_train, y_test = partition_data(df = df, split_by = SPLIT_BY,
                                                      x="employer_description", y = "industry", validation_set=False,
                                                      test_size=0.15, random_state=8082023)
    print("Number of training samples: %d" %len(x_train))
    print("Number of testing samples: %d" %len(x_test))

    # load encoder, tokenizer and model
    le = get_label_encoder(y = df["industry"])
    tokenizer, model = load_xlm_pretrained(
        path_to_model = "../hf_models/xlm-roberta-base", 
        n_targets = len(le.classes_)
        )

    # tokenize the data
    x_train = tokenizer(x_train.to_list(), return_tensors="pt", truncation=True, max_length=128, padding=True)
    x_test = tokenizer(x_test.to_list(), return_tensors="pt", truncation=True, max_length=128, padding=True)
    y_train, y_test = encode_labels(label_encoder = le, y = y_train), encode_labels(label_encoder=le, y = y_test) 

    # data pipelines
    train_dl = DataLoader(
        IndustryClassificationDataset(targets = y_train, texts = x_train),
        batch_size = BATCH_SIZE, shuffle = True
        )
    test_dl = DataLoader(
        IndustryClassificationDataset(targets = y_test, texts = x_test),
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
        model = model, n_epochs = EPOCHS, optimizer=optim,
        train_loader = train_dl, eval_loader = test_dl, 
        device= DEVICE, return_finetuned_model=True
        )
    
    # model report:
    report = get_model_report(model = finetuned_model, eval_loader = test_dl,
                              label_encoder = le, device = DEVICE)
    print(report)

    # retrain model an complete dataset and save model:
    if SAVE_MODEL_PATH:
        tokenizer, model = load_xlm_pretrained(
            path_to_model = "../hf_models/xlm-roberta-base", 
            n_targets = len(le.classes_)
            )
        x_train = tokenizer(df["employer_description"].to_list(), return_tensors="pt", truncation=True, max_length=128, padding=True)
        y_train = encode_labels(label_encoder = le, y = df["industry"])
        train_dl = DataLoader(
            IndustryClassificationDataset(targets = y_train, texts = x_train),
            batch_size = BATCH_SIZE, shuffle = True)
        finetuned_model = finetune(model = model, n_epochs = EPOCHS, train_loader = train_dl, optimizer=optim, 
                                   silent_training=True, device= DEVICE, return_finetuned_model=True)
        torch.save(finetuned_model, SAVE_MODEL_PATH)
        print("Model saved at: ", SAVE_MODEL_PATH)
