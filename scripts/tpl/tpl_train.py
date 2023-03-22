import torch
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

from jpap.tpl.endetect import *

#### configure:
DAT_PATH = "C:/Users/matth/Documents/cieb/endetect_train.csv"
MAX_TOKENS = 60000
MAX_LEN = 50
BATCH_SIZE = 64
EPOCHS = 16

#### load and label the data:------------------------------
loader = ENDetectionTrainLoader(dataset_path= DAT_PATH, target= "en", target_share = 0.25)\
    .vocab(max_tokens = MAX_TOKENS, partition=None)\
    .split(test_size=0.15)

# training data
train_df = loader.getLangDetectDataset(partition = "train", output_mode = "pt", max_len = MAX_LEN)
train_dl = DataLoader(dataset = train_df, batch_size = BATCH_SIZE, shuffle = True) # define the pipeline for training

# validation data
y_valid = loader.label(partition="validation", ouput_mode="pt")
x_valid = loader.tokenize_sequence(partition="validation", output_mode="pt")

# testing data
y_test = loader.label(partition="test", ouput_mode="pt")
x_test = loader.tokenize_sequence(partition="test", output_mode="pt")

#### setting up the model, define the loss and optimizer: ------------------------------
model = ENDetectionModel(vocab_size = MAX_TOKENS, embedding_dim = 64, hidden_units = 128)
loss_function = torch.nn.BCELoss()
model_optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

#### train the english detection model------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("------------ Training on %s --------------" % device)
torch.manual_seed(20032023)

loss_hist_train, acc_hist_train, loss_hist_val, acc_hist_val = ENDetectTrain(
    model = model, epochs = EPOCHS, 
    train_dl=train_dl, x_valid=x_valid, y_valid=y_valid, 
    model_optimizer=model_optimizer, loss_function=loss_function
    )

# check performance 
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1,2,1)
plt.plot(loss_hist_train, lw=4)
plt.plot(loss_hist_val, lw=4)
plt.legend(["train loss", "validation loss"])
ax.set_label("epochs")
ax = fig.add_subplot(1,2,2)
plt.plot(acc_hist_train, lw=4)
plt.plot(acc_hist_val, lw=4)
plt.legend(["train accuracy", "validation accuracy"])
ax.set_label("epochs")
plt.show()

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

#### saving the model ------------------------------
# model._save_to_state_dict(SAVE_PATH)