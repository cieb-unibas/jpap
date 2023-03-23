import torch
from torch.utils.data import DataLoader

from jpap.tpl.endetect import *

#### configure:
DAT_PATH = "C:/Users/matth/Documents/cieb/endetect_train.csv"
MAX_TOKENS = 50000
MAX_LEN = 30
BATCH_SIZE = 16
EPOCHS = 11

#### load and label the data:------------------------------
processor = ENDetectionProcessor(dataset_path= DAT_PATH, target= "en") # initialize
vocab = processor.vocab(max_tokens = MAX_TOKENS, texts = processor.dataset["text"]).tokenizer.vocabulary # set and extract the vocabulary
processor.dataset = processor.get_data(target_share = 0.25) # downsample the data to give more weight to english
processor = processor.split(test_size=0.15) # split to train-test-val.

# training data
train_dl = DataLoader(
    dataset = processor.getLangDetectDataset(partition = "train", output_mode = "pt", max_len = MAX_LEN), 
    batch_size = BATCH_SIZE, shuffle = True
    )

# validation data
y_valid = processor.label(partition="validation", ouput_mode="pt")
x_valid = processor.tokenize_sequence(partition="validation", output_mode="pt", max_len = MAX_LEN)

# testing data
y_test = processor.label(partition="test", ouput_mode="pt")
x_test = processor.tokenize_sequence(partition="test", output_mode="pt", max_len = MAX_LEN)

#### setting up the model, define the loss and optimizer: ------------------------------
model = ENDetectionModel(vocab_size = MAX_TOKENS, embedding_dim = 64, hidden_units = 128)
loss_function = torch.nn.BCELoss()
model_optimizer = torch.optim.SGD(model.parameters(), lr = 0.005, momentum=0.9)

#### train the english detection model------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("------------ Training on %s --------------" % device)
torch.manual_seed(20032023)

loss_hist_train, acc_hist_train, loss_hist_val, acc_hist_val = ENDetectTrain(
    model = model, epochs = EPOCHS, 
    train_dl=train_dl, x_valid=x_valid, y_valid=y_valid, 
    model_optimizer=model_optimizer, loss_function=loss_function
    )

#### testing the model ------------------------------
def evaluate(model, x, y, loss_function):
    # predict
    pred = model(x)[:, 0]
    pred = pred.reshape(pred.shape[0])
    # loss
    loss = loss_function(pred, y.float()).item()
    # accuracy
    correct = ((pred>=0.5).float() == y).float()
    acc = correct.mean().float().item()
    return round(loss, 3), round(acc,3)

test_loss, test_acc = evaluate(model=model, x = x_test, y = y_test, loss_function=loss_function)
print("""------ Test loss: %f ------
      \n------ Test accuracy: %f ------
      """ % (test_loss, test_acc))

#### inference:
tokenizer = ENDetectTokenizer()
tokenizer.load_vocabulary(vocab)
example_sentence = "tomorrow i need to go to the hospital because i am taking my wisdom teeth out"
tokenized_sentence = tokenizer.tokenize(text = example_sentence, sequence_length = MAX_LEN, padding_idx = 0)
for n in tokenized_sentence:
    [{n: k} for k in tokenizer.vocabulary.keys() if tokenizer.vocabulary[k] == n]
tokenized_sentence = torch.tensor([tokenized_sentence], dtype=torch.int32)
print("The sentence '%s' is english: " % example_sentence, (model(tokenized_sentence)[:, 0] >= 0.5).item())

#### saving the model ------------------------------
# model._save_to_state_dict(SAVE_PATH)