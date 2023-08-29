# jpap
jpap ('job postings analyses pipelines') contains different pipelines for in-depth analysis of job postings.

## To-Do's
- finetune pre-trained model on industry dataset
- translation pipeline using fine-tuned encoder-decoder architecture
- ...

## Installation

#### Step 1
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

#### Step 2
Check for GPU support on your machine and setup `PyTorch` according to the [developer website](https://pytorch.org/).

#### Step 3
Install the `transformers` library as specified by [Huggingface](https://huggingface.co/docs/transformers/installation):

```bash
conda install -c huggingface transformers
```

## Job Postings Industry Pipeline (IPL)


### How?
- IPL **leverages pre-trained language models** for feature extraction and classification. Feature extraction is based on the `multilingual-MiniLMv2-L6-mnli-xnli` model, a version of `XLM-RoBERTa-large` finetuned on the `XNLI` and `MNLI` datasets for zero-shot-classification. 

- IPL then follows a **weak supervision** approach with **labelling-fuctions** (LF) to label a subset of postings to industries. It does this based on companies, for which the ground truth is known (e.g. roche is from the pharmaceutical sector) as well as on certain patterns in the company name that can be unambigously assigned to specific industries (e.g. 'university', 'hospital' or 'restaurant').

- Since postings of the same companies can be very similar, **sampling** procedure is an issue because of potential data leakage when splitting training and test set. For example, a job posting of Roche could land in the training and another one in the testing set even though they are (practically) identical. For the IPL training the dataset was does split by employers so that all postings from a certain employer can either be in the test or in the validation set. Furthermore, company names are blinded for the classifier so that it does not classify postings to industries based on company names.

- The IPL classifier is based on **transfer learning** as it is a fine-tuned version of the `XLM-RoBERTa` model, which in itself is a multilingual version of `RoBERTa` model that has been pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages by researchers at Facebook AI. 

## Jop Postings Translation Pipeline (TPL)
Multilingual translations for job postings.