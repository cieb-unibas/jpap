import string

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from datasets import load_dataset

class ENDetectionTrainLoader():
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

    def label(self, partition, stratified_dataset = False, ouput_mode = "np"):
        self.label_dict = {"rest": 0, self.target: 1}
        if stratified_dataset:
            out = [1 if x == self.target else 0 for x in self.stratified_dataset[partition]["labels"]]           
        else:
            out = [1 if x == self.target else 0 for x in self.dataset[partition]["labels"]]
        if ouput_mode == "pt":
            return torch.tensor(out, dtype=torch.int)
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

    def tokenize_sequence(self, partition: str, output_mode: str, stratified_data = False, max_len: int = None):
        if stratified_data:
            tokenized_sequence = [self.tokenize(text) for text in self.stratified_dataset[partition]["text"]]
        else:
            tokenized_sequence = [self.tokenize(text) for text in self.dataset[partition]["text"]]
        if max_len == None:
            max_len = max([len(x) for x in tokenized_sequence])
        tokenized_sequence = [x if len(x) <= max_len else x[:max_len] for x in tokenized_sequence]
        tokenized_sequence = [x + ([0] * (max_len - len(x))) for x in tokenized_sequence]
        if output_mode == "pt":
            return torch.tensor(tokenized_sequence, dtype=torch.int)
        else:
            return tokenized_sequence
        
    def stratify(self, target_share: float = 0.3):
        """
        Stratify every partition to have at least `target_share` of target samples.
        """
        for partition in ["train", "test", "validation"]:
            # get the data
            if self.source == "huggingface":
                df = pd.DataFrame(self.dataset[partition])
            else:
                df = self.dataset[partition]
            # split data by target labels
            target_class, rest = df.loc[(df.labels == self.target), :].reset_index(drop=True), df.loc[(df.labels != self.target), :].reset_index(drop=True) 
            n_rest = round(len(target_class) / target_share)
            # subset, concatate and shuffle
            df = pd.concat([target_class, rest.sample(n = n_rest, random_state=self.seed)])
            df = df.sample(frac = 1, random_state = self.seed).reset_index(drop = True)
            # update the original data
            self.stratified_dataset = {partition: {"labels": [], "texts": []}}
            self.stratified_dataset[partition]["labels"] += df["labels"]
            self.stratified_dataset[partition]["texts"] += df["texts"]
            print("%s dataset stratified." % partition)

    def getLangDetectDataset(
            self, partition: str = "train", stratified_dataset = False,
            output_mode: str = "pt", max_len: int = 100):
        """
        Transform and extracts a torch Dataset from the data.
        """
        if stratified_dataset:
            df = ENDetectDataset(
                labels = self.label(partition=partition, stratified_dataset=True, ouput_mode=output_mode),
                texts = self.tokenize_sequence(partition=partition, stratified_dataset=True, output_mode=output_mode, max_len=max_len)
                )
        else:
            df = ENDetectDataset(
                labels = self.label(partition=partition, ouput_mode=output_mode),
                texts = self.tokenize_sequence(partition=partition, output_mode=output_mode, max_len=max_len)
                )
        return df       

class ENDetectDataset(Dataset):
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

class ENDetectionModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, sequence_length : int) -> None:
        super(ENDetectionModel, self).__init__()
        # parameters:
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        # layers
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim = embedding_dim)
        self.L1 = nn.Linear(embedding_dim, int(embedding_dim / 4), bias=True)
        a1 = nn.ReLU()
        self.L2 = nn.Linear(int(embedding_dim / 4), 1, bias=True)
        a2 = nn.Sigmoid()
        self.init_weights()
        self.model_list = nn.ModuleList([self.embedding, self.L1, a1, self.L2, a2])

    def init_weights(self, init_range: float = 0.5):
        init_range = abs(init_range)
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.L1.weight.data.uniform_(-init_range, init_range)
        self.L1.bias.data.zero_()
        self.L2.weight.data.uniform_(-init_range, init_range)
        self.L2.bias.data.zero_()

    def forward(self, x):
        for f in self.model_list:
            x = f(x)
        return x

def ENDetectTrain(model, epochs, train_dl, x_valid, y_valid, model_optimizer, loss_function, train_samples):
    # initialize the losses and accuracies
    loss_hist_train = [0] * epochs
    acc_hist_train = [0] * epochs
    loss_hist_val = [0] * epochs
    acc_hist_val = [0] * epochs

    for epoch in range(epochs):
        # train on batches
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0] # predict
            pred = pred.reshape(pred.shape[0])
            loss = loss_function(pred, y_batch.float()) # calculate loss
            loss.backward() # calculate the gradient on the loss
            model_optimizer.step() # get the optimizer
            model_optimizer.zero_grad() # set optimizer weights to zero again
            loss_hist_train[epoch] += loss.item() # record the loss
            correct = ((pred>=0.5).int() == y_batch).float() # record the accuracy
            acc_hist_train[epoch] += correct.mean().float().item()
        
        # get the mean training loss and accuracy across batches per epoch
        loss_hist_train[epoch] /= train_samples
        acc_hist_train[epoch] /= train_samples/len(y_batch)
        
        # preditc and evaluate on validation set
        pred = model(x_valid)[:, 0]
        pred = pred.reshape(pred.shape[0])
        loss = loss_function(pred, y_valid.float())
        loss_hist_val[epoch] += loss.item()
        correct = ((pred>=0.5).int() == y_valid).float()
        acc_hist_val[epoch] += correct.mean().float().item()

        # log:
        if epoch % 5 == 0:
            print("Validation loss after %d epochs: " % epoch, round(loss_hist_val[epoch], 3),
                  "\nValidation accuracy after %d epochs: " % epoch, round(acc_hist_val[epoch], 3))


    return loss_hist_train, acc_hist_train, loss_hist_val, acc_hist_val

