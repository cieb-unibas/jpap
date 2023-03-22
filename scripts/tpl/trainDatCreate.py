from datasets import load_dataset
import pandas as pd

save_path = "C:/Users/matth/Documents/cieb/"
dat = load_dataset(path = "papluca/language-identification")

df = pd.DataFrame()
for partition in dat:
    df = pd.concat([df, dat[partition].to_pandas()], axis = 0)

df.reset_index(drop=True, inplace=True)
df.to_csv(save_path + "endetect_train.csv", header=True, index=False)
print("Training data saved on disk")
