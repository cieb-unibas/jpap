from torch.utils.data import DataLoader

from jpap.tpl import DatasetLoader, LangDetectDataset

#### load and label the data:
loader = DatasetLoader(source = "huggingface", dataset_id = "papluca/language-identification", target= "en").vocab(max_tokens=50000, partition="train")
# loader.vocab(max_tokens=50000, partition="train")
# x_train = loader.tokenize_sequence(partition="train", output_mode="pt", max_len=100)
# y_train = loader.label(partition="train", ouput_mode="pt")
# train_df = LDetectDataset(labels=y_train, texts=x_train)
train_df = LangDetectDataset(
    labels = loader.label(partition="train", ouput_mode="pt"),
    texts = loader.tokenize_sequence(partition="train", output_mode="pt", max_len=100)
    )
train_dl = DataLoader(dataset = train_df, batch_size = 128, shuffle = True)
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html


# b) using xlm-roberta language specific tokens ------------
# https://colab.research.google.com/drive/15LJTckS6gU3RQOmjLqxVNBmbsBdnUEvl?usp=sharing#scrollTo=V_gbHRmNHEWU
# https://huggingface.co/xlm-roberta-base
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
tokenizer(data.dataset["train"]["text"][1], return_tensors = "pt")


