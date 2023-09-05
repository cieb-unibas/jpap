# JPAP
The idea of JPAP ('job postings analyses pipelines') is to create different pipelines for in-depth analysis of job postings contained in the CIEB Job Postings Database (JPOD).

## Installation

`cd` into this repository, setup a virtual environment via and install dependencies:

Using conda
```bash
conda create --name jpap python=3.9
conda install -r requirements.txt
```

Using pip
```bash
python venv jpap-venv
pip install --upgrade pip
pip install -r requirements.txt
```

#### Troubleshooting
If you encounter problems installing the dependencies for JPAP, check for GPU support on your machine and setup `PyTorch` manually according to the [developer website](https://pytorch.org/). Consider doing the same  for the `transformers` library as specified by [Huggingface](https://huggingface.co/docs/transformers/installation):


## Job Postings Industry Pipeline (IPL)

### How?
- IPL **leverages pre-trained language models** for feature extraction and classification. Feature extraction is based on the `multilingual-MiniLMv2-L6-mnli-xnli` model, a version of `XLM-RoBERTa-large` finetuned on the `XNLI` and `MNLI` datasets for zero-shot-classification. 

- IPL then follows a **weak supervision** approach with **labelling-fuctions** (LF) to label a subset of postings to industries. It does this based on companies, for which the ground truth is known (e.g. roche is from the pharmaceutical sector) as well as on certain patterns in the company name that can be unambigously assigned to specific industries (e.g. 'university', 'hospital' or 'restaurant').

- Since postings of the same companies can be very similar, **sampling** procedure is an issue because of potential data leakage when splitting training and test set. For example, a job posting of Roche could land in the training and another one in the testing set even though they are (practically) identical. For the IPL training the dataset was does split by employers so that all postings from a certain employer can either be in the test or in the validation set. Furthermore, company names are blinded for the classifier so that it does not classify postings to industries based on company names.

- The IPL classifier is based on **transfer learning** as it is a fine-tuned version of the `XLM-RoBERTa` model, which in itself is a multilingual version of `RoBERTa` model that has been pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages by researchers at Facebook AI. 

## Jop Postings Translation Pipeline (TPL)
Multilingual translations for job postings.

- Work in Progress
- Consider using a translation pipeline using fine-tuned encoder-decoder architecture: Check the following [paper](https://arxiv.org/pdf/2010.11125.pdf), [code][https://github.com/kadirnar/Multilingual-Translation] and [model][https://huggingface.co/facebook/m2m100_418M] from Facebook AI.
