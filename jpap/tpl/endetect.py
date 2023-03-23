import string

import pandas as pd
import random

import torch
import torch.nn as nn

from torch.utils.data import Dataset

class ENDetectionProcessor():
    def __init__(
            self, dataset_path: str = "C:/Users/matth/Documents/cieb/endetect_train.csv", 
            target: str = "en", target_share : float = None, seed: int = 10032023
            ):
        self.dataset_path = dataset_path
        self.target = target
        self.target_share = target_share
        self.seed = seed
        self.dataset = self.get_data()
        self.tokenizer = ENDetectTokenizer()
    
    def get_data(self, target_share = None):
        dat = pd.read_csv(filepath_or_buffer = self.dataset_path)
        if target_share:
            self.target_share = target_share
        if self.target_share:
            df_target_group = dat.loc[dat.labels == self.target, :].reset_index(drop=True)
            n_target_group = len(df_target_group)
            df_other_group = dat.loc[dat.labels != self.target, :].reset_index(drop=True)
            n_other_group = round((1-target_share) * (n_target_group / target_share))
            random.seed(self.seed)
            df_other_group = df_other_group.iloc[random.choices(range(len(df_other_group)), k=n_other_group), :].reset_index(drop=True)
            dat = pd.concat([df_target_group, df_other_group], axis = 0)
        return dat
    
    def split(self, test_size):
        assert isinstance(self.dataset, pd.DataFrame), "Convert dataset to a pandas.DataFrame object."
        if isinstance(test_size, float):
            test_size = round(len(self.dataset) * test_size)
            n_val = test_size
        else:
            n_val = test_size
        random.seed(self.seed)
        self.dataset = self.dataset.sample(frac = 1).reset_index(drop = True) # shuffle
        df = {}
        df["test"] = self.dataset.iloc[:test_size, :].reset_index(drop=True)
        df["validation"] = self.dataset.iloc[test_size : test_size + n_val, :].reset_index(drop=True)
        df["train"] = self.dataset.iloc[test_size + n_val:, :].reset_index(drop=True)
        self.dataset = df
        return self

    def label(self, partition, ouput_mode = "np"):
        self.label_dict = {"other": 0, self.target: 1}
        out = [1 if x == self.target else 0 for x in self.dataset[partition]["labels"]]
        if ouput_mode == "pt":
            return torch.tensor(out, dtype=torch.int)
        else:
            return out

    def vocab(self, max_tokens, texts):
        self.tokenizer.set_vocabulary(max_tokens=max_tokens, texts = texts)
        return self

    def tokenize_sequence(self, partition: str, output_mode: str, max_len: int = None):
        tokenized_sequence = [self.tokenizer.tokenize(text) for text in self.dataset[partition]["text"]]
        if max_len == None:
            max_len = max([len(x) for x in tokenized_sequence])
        tokenized_sequence = [x if len(x) <= max_len else x[:max_len] for x in tokenized_sequence]
        tokenized_sequence = [x + ([0] * (max_len - len(x))) for x in tokenized_sequence]
        if output_mode == "pt":
            return torch.tensor(tokenized_sequence, dtype=torch.int)
        else:
            return tokenized_sequence
        
    def getLangDetectDataset(self, partition: str = "train", output_mode: str = "pt", max_len: int = 100):
        """
        Transform and extracts a torch Dataset from the data.
        """
        df = ENDetectDataset(
            labels = self.label(partition=partition, ouput_mode=output_mode),
            texts = self.tokenize_sequence(partition=partition, output_mode=output_mode, max_len=max_len)
            )
        return df       

class ENDetectTokenizer():
    def __init__(self) -> None:
        return

    def split_to_tokens(self, text):
        cleaned_text = "".join([c for c in text if c not in string.punctuation and c not in string.digits])
        tokenized_text = cleaned_text.lower().split()
        return tokenized_text
    
    def load_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary
   
    def set_vocabulary(self, texts, max_tokens: int = 50000):
        self.max_tokens = max_tokens
        counter = {}
        for text in texts:
            tokens = self.split_to_tokens(text)
            for token in tokens:
                if token not in counter:
                    counter[token] = 0
                counter[token] += 1
        sorted_tokens = sorted(counter.items(), key = lambda x: x[1], reverse=True)
        sorted_tokens = [c[0] for c in sorted_tokens]
        if self.max_tokens:
            sorted_tokens = sorted_tokens[:self.max_tokens-2]
        sorted_tokens = ["<pad>", "<unk>"] + sorted_tokens
        self.vocabulary = {token: i for i, token in enumerate(sorted_tokens)}
        print("Vocabulary specified")
    
    def tokenize(self, text, sequence_length: int = None, padding_idx: int = 0):
        processed_text = [self.vocabulary[t] if t in self.vocabulary.keys() else 1 for t in self.split_to_tokens(text)]
        if sequence_length != None:
            if len(processed_text) > sequence_length:
                processed_text = processed_text[:sequence_length]
            else:
                n_padding = sequence_length - len(processed_text)
                processed_text = processed_text + ([padding_idx] * n_padding)
        return processed_text

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
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_units: int) -> None:
        super(ENDetectionModel, self).__init__()
        # parameters:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # layers
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim = embedding_dim, padding_idx=0)
        self.L1 = nn.Linear(embedding_dim, hidden_units, bias=True)
        a1 = nn.ReLU()
        self.L2 = nn.Linear(hidden_units, int(hidden_units / 4), bias=True)
        a2 = nn.ReLU()
        self.L3 = nn.Linear(int(hidden_units / 4), 1, bias=True)
        a3 = nn.Sigmoid()
        self.init_weights()
        self.model_list = nn.ModuleList([self.embedding, self.L1, a1, self.L2, a2, self.L3, a3])

    def init_weights(self, init_range: float = 0.5):
        init_range = abs(init_range)
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.L1.weight.data.uniform_(-init_range, init_range)
        self.L1.bias.data.zero_()
        self.L2.weight.data.uniform_(-init_range, init_range)
        self.L2.bias.data.zero_()
        self.L3.weight.data.uniform_(-init_range, init_range)
        self.L3.bias.data.zero_()

    def forward(self, x):
        for f in self.model_list:
            x = f(x)
        return x
    
    def predict(self, x):
        pred = self(x)[:, 0]
        return pred


def ENDetectTrain(model, epochs, train_dl, x_valid, y_valid, model_optimizer, loss_function):
    # initialize the losses and accuracies
    loss_hist_train = [0] * epochs
    acc_hist_train = [0] * epochs
    loss_hist_val = [0] * epochs
    acc_hist_val = [0] * epochs

    for epoch in range(epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0] # predict
            pred = pred.reshape(pred.shape[0])
            loss = loss_function(pred, y_batch.float()) # calculate loss
            loss.backward() # calculate the gradient on the loss
            model_optimizer.step() # get the optimizer
            model_optimizer.zero_grad() # set optimizer weights to zero again
            
            # store the loss on this batch
            loss_hist_train[epoch] += loss.item()
            
            # calculate and store the accuracy on this batch
            correct = ((pred>=0.5).float() == y_batch).float()
            acc_hist_train[epoch] += correct.mean().float().item()
        
        # get the average training loss and accuracy per batch in this epoch
        loss_hist_train[epoch] /= len(train_dl)
        acc_hist_train[epoch] /= len(train_dl)
        
        # preditc and evaluate on validation set
        pred = model(x_valid)[:, 0]
        pred = pred.reshape(pred.shape[0])
        loss = loss_function(pred, y_valid.float())
        loss_hist_val[epoch] += loss.item()
        correct = ((pred>=0.5).float() == y_valid).float()
        acc_hist_val[epoch] += correct.mean().float().item()

        # log:
        if epoch % 5 == 0:
            print("Validation loss after %d epochs: " % epoch, round(loss_hist_val[epoch], 3),
                  "\nValidation accuracy after %d epochs: " % epoch, round(acc_hist_val[epoch], 3))


    return loss_hist_train, acc_hist_train, loss_hist_val, acc_hist_val