#### train the language detection model necessary for the translation pipeline (tpl)
import numpy as np
import torch

from jpap.tpl import DatasetLoader

# data:
data = DatasetLoader(source = "huggingface", dataset_id = "papluca/language-identification", targets=["en"])
data.update()
data = data.dataset

