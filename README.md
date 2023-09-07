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
JPOD does not feature any (reliable) information about the industries that job postings are associated with. The industry pipeline (IPL) is an attempt to solve this problem. IPL is powered by **pre-trained language models**, which it uses for feature extraction and classification (see explanations below). 

IPL can be configured to assign postings to **four different industry levels: 'nace' 'meso', 'macro' and 'pharma'**. The corresponding industry mappings can be found in the ["data/"](./data/raw/) directory and performance reports of the trained models can be seen in the following ["logs/"](./scripts/ipl/logs/) directory. An overview is given in the table below.

industry level|# industries|weighted F1
---|---|---
nace|18|68%
meso|13|64%
macro|6|73%
pharma|2|89%

### How to use IPL?

:warning: **IMPORTANT:** `IPL` makes use of two relatively large language models that are computationally expensive to use. Hence, you are strongly recommended to use the scicore cluster's GPUs via a slurm script (see <a href="./examples/main_scicore.sh">here</a> for an example).

After you `cd` into this repository's directory, `IPL` can easily be loaded as follows:

```python
from jpap.ipl import IPL

industry_pipeline = IPL(classifier = "macro")
```
:exclamation: **A more extensive example highlighting how to use `IPL` together with `jpod` can be found  <a href='./examples/'>here</a>.**

### How was IPL trained and how does it work?
To train `IPL`, two particular challenges have to be apporached: The lack of labelled training data and a low signal-to-noise ratio in postings text.

The first refers to....

The second aspect refers to the fact that a substantial part of job postings' texts is not related to the company that published the particular job opening. Rather it describes tasks and responsibilities, which can be very similar across industries. Think, for example, about a secratary in the pharmaceutical industry or within a law firm: Both of them have likely similar tasks. To differentiate industries, this text snippets do not contain any signal for a classifier and should thus be excluded. `IPL` implements a strategy to accomplish that, which is based on NLP and a pre-trained language model. 

Once the (potentially) relevant parts of the text are identified, `IPL` uses a second language model on top for classification of job postings to industries. 

Below are more comprehensive explanations.

**Training Data**: IPL follows a **weak supervision** approach with **labelling-fuctions** to label a subset of postings to industries. It does this based on companies, for which the ground truth is known (e.g. roche is from the pharmaceutical sector) as well as on certain patterns in the company name that can be unambigously assigned to specific industries (e.g. 'university', 'hospital' or 'restaurant').

Since postings of the same companies can be very similar, **sampling** procedure is an issue because of potential data leakage when splitting training and test set. For example, a job posting of Roche could land in the training and another one in the testing set even though they are (practically) identical. For the IPL training the dataset was does split by employers so that all postings from a certain employer can either be in the test or in the validation set. Furthermore, company names are blinded for the classifier so that it does not classify postings to industries based on company names.

**Feature extraction:** is based on combining two strategies. First, all sentences in a job posting that mention the employer's name are considered relevant and extracted using keyword searches. Second, a zero-shot classification model is used to extract additional relevant sentences from the remaining text. For this purpose `IPL` builds on the <a href="https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli">`multilingual-MiniLMv2-L6-mnli-xnli`</a> transformer model, which is a finetuned version of <a href="https://huggingface.co/xlm-roberta-large">`XLM-RoBERTa-large`</a> for multilingual zero-shot-classification (trained on the `XNLI` and `MNLI` datasets). Every sentence of a particular job posting is sent to this model, and if the model classifies it as a description of "who we are", "who this is" or an "industry or sector", it is also extracted.

**Classification:** The IPL classifier is based on **transfer learning** as it is a fine-tuned version of the `XLM-RoBERTa` model, which in itself is a multilingual version of `RoBERTa` model that has been pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages by researchers at Facebook AI. 


## Jop Postings Translation Pipeline (TPL)
Multilingual translations for job postings.

- Work in Progress
- Consider using a translation pipeline using fine-tuned encoder-decoder architecture: Check the following [paper](https://arxiv.org/pdf/2010.11125.pdf), [code][https://github.com/kadirnar/Multilingual-Translation] and [model][https://huggingface.co/facebook/m2m100_418M] from Facebook AI.
