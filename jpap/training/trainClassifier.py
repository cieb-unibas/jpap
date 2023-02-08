import os

from transformers import AutoTokenizer, AutoModel
import pandas as pd

#### extract relevant sentences:


#### model configuration --------------------
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

#### tokenization ------------------

# tokenize the texts:
inputs = tokenizer(list(df["job_description"][0]), padding = "max_length", truncation = True, return_tensors="pt")
# this is equivalent to running:
# tokens = tokenizer.tokenize(sequence)
# inputs = tokenizer.convert_tokens_to_ids(tokens)
# => und dann sequences mit [101] rsp. [102] verlängern.

# get the model predictions:
outputs = model(**inputs)
outputs
f'Model has {outputs.last_hidden_state.shape[2]} dimension in last hidden layer'