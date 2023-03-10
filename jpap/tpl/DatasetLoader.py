import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

class DatasetLoader():
    def __init__(self, source: str = "huggingface", dataset_id: str = "papluca/language-identification", targets: list = ["en"], seed: int = 10032023):
        self.source = source
        self.dataset_id = dataset_id
        self.dataset = self.get_data()
        self.targets = targets
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
        labelled = [l if l in self.targets else "rest" for l in data["labels"]]
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