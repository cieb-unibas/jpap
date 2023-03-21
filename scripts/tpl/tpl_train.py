import torch
from torch.utils.data import DataLoader
from jpap.tpl.endetect import *

#### configure:
MAX_TOKENS = 50000
MAX_LEN = 50
BATCH_SIZE = 64
EPOCHS = 5

#### load and label the data:
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
loader = ENDetectionTrainLoader(source = "huggingface", dataset_id = "papluca/language-identification", target= "en")\
    .vocab(max_tokens = MAX_TOKENS, partition = "train") # generate a vocabulary for tokenizing the text
loader.stratify(target_share=0.25)

# check WeightedRandomSampler() from torch.utils.data or other 

train_df = loader.getLangDetectDataset(partition = "train", output_mode = "pt", max_len = MAX_LEN) # get the torch.Dataset for the training partition for this data
train_dl = DataLoader(dataset = train_df, batch_size = BATCH_SIZE, shuffle = True) # define the pipeline for training
y_valid = loader.label(partition="validation", ouput_mode="pt")
x_valid = loader.tokenize_sequence(partition="validation", output_mode="pt")

#### setting up the model, loss and optimizer:
# https://coderzcolumn.com/tutorials/artificial-intelligence/word-embeddings-for-pytorch-text-classification-networks
model = ENDetectionModel(vocab_size=MAX_TOKENS, embedding_dim=64, sequence_length=MAX_LEN)
loss_function = torch.nn.BCELoss()
model_optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#### train the english detection model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("------------ Training on %s --------------" % device)
torch.manual_seed(20032023)
ENDetectTrain(
    model = model, epochs = EPOCHS, train_dl=train_dl, 
    x_valid=x_valid, y_valid=y_valid, model_optimizer=model_optimizer,
    loss_function=loss_function, train_samples=len(train_df)
    )

#### testing the model


#### saving the model


# b) using xlm-roberta language specific tokens ------------
# https://colab.research.google.com/drive/15LJTckS6gU3RQOmjLqxVNBmbsBdnUEvl?usp=sharing#scrollTo=V_gbHRmNHEWU
# https://huggingface.co/xlm-roberta-base
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
# tokenizer(data.dataset["train"]["text"][1], return_tensors = "pt")