# JPAP [WORK IN PROGRESS]
The idea of JPAP ('job postings analyses pipelines') is to create different pipelines for in-depth analysis of job postings contained in the CIEBs Job Postings Database (<a href="https://github.com/cieb-unibas/jpod">JPOD</a>).

## Usage

You can either use JPAP via an existing virtual environment on the scicore cluster or setup a new one yourself for your project.

### Use existing scicore virtual environment
To activate the jpap environment on scicore run the following on the command line:

```bash
cd /scicore/home/weder/GROUP/Innovation/05_job_adds_data/
source jpap-venv/bin/activate
```

### Install a new virtual environment manually
To install a new environment, `cd` into this repository, setup a virtual environment via conda or pip and install dependencies:

Using conda
```bash
conda create --name YOUR_ENV_NAME python=3.9
conda install -r requirements.txt
```

Using pip
```bash
python venv YOUR_ENV_NAME
pip install --upgrade pip
pip install -r requirements.txt
```

#### Troubleshooting
If you encounter problems installing the dependencies for JPAP, check for GPU support on your machine and setup `PyTorch` manually according to the [developer website](https://pytorch.org/). Consider doing the same  for the `transformers` library as specified by [Huggingface](https://huggingface.co/docs/transformers/installation):


## Job Postings Industry Pipeline (IPL)

### Challenge and Approach
JPOD does not feature any (reliable) information about the industries that job postings are associated with. The industry pipeline (IPL) is an attempt to solve this problem. IPL is powered by **pre-trained language models**, which it uses for feature extraction from job postings' texts and classification thereof (see explanations below). 

IPL can be configured to assign job postings (and thereby companies) to **four different industry levels: 'nace' 'meso', 'macro' and 'pharma'**. The corresponding industry mappings can be found in the [`data/raw/`](./data/raw/) directory and performance reports of the corresponding trained models can be accessed in the following [`scirpts/ipl/logs/`](./scripts/ipl/logs/) directory. An overview is given in the table below.

industry level|# industries|weighted F1
---|---|---
nace|18|68%
meso|13|64%
macro|6|73%
pharma|2|89%

### How to use IPL?

:warning: **IMPORTANT:** `IPL` makes use of two relatively large language models that are computationally expensive to use. Hence, you are strongly recommended to use the scicore cluster's GPUs via a slurm script (see <a href="./examples/ex_scicore.sh">here</a> for an example).

After you `cd` into this repository's directory, `IPL` can easily be loaded and applied to job postings texts:

```python
import sys
import os

sys.path.append(os.getcwd())

from jpap.ipl import IPL

# helper function to load and predict sample
def load_and_predict_example(company_name, pipeline):
    with open(f'./examples/retrieved_postings/{company_name}.txt', "r") as f:
        text = f.read().replace("\n", " ")
    industry_label = pipeline(postings = [text], company_names = [company_name])
    print(f'"{company_name} is predicted to be part of the industry: "{industry_label[0]}""')

# load and specify IPL
industry_pipeline = IPL(classifier = "macro")

# inference with an example posting from the pharmaceutical industry: https://acino.swiss/
company_name = "acino"
load_and_predict_example(company_name, industry_pipeline)
"acino is predicted to be part of the industry: "pharmaceutical and life sciences""

# inference with an example posting from the financial industry: https://www.saanenbank.ch/de
company_name = "saanen_bank"
load_and_predict_example(company_name, industry_pipeline)
"saanen_bank is predicted to be part of the industry: "services""
```

:exclamation: **A more extensive example highlighting how to use `IPL` together with `jpod` can be found  <a href='./examples/'>here</a>.**


### How was IPL trained and how does it work?
To build `IPL`, two particular challenges had to be apporached: First, the lack of labelled training data and second, a low signal-to-noise ratio in postings text.

#### Data Collection
To train a text classification model, one needs labelled training samples. In case of `IPL` these would correspond to job postings that are labelled to industries (e.g., a job posting from Novartis should be labelled as belonging to the pharmaceutical industry). Such data is not available directly. To construct it, `IPL` follows a **weak supervision** approach with **labelling-fuctions** to label a subset of postings to industries. 

It does this based on two different heuristics: First it assigns certain employers to industries, for which the ground truth is known (e.g. Novartis is from the pharmaceutical industry). Second, it leverages that certain employers have patterns in their name that can be unambigously assigned to specific industries (e.g. employers with one of the patterns 'university', 'hospital' or 'restaurant' can clearly be assigned to the industries 'academia', 'health' or 'restaurants and accomodation', respectively). 
Postings from employers that are labelled according to the above-mentioned approaches are then retrieved from the CIEBs Job Postings Database (<a href="https://github.com/cieb-unibas/jpod">JPOD</a>) and used for training. For details regarding the construction of this training sample, see the correspodning [employer mappings](./data/raw/industry_label_companies.json), [industry mappings](./data/raw/industry_label_patterns.json) and [this script](./scripts/ipl/ipl_dat.py) for the implementation.

**Note** that naturally, you can adapt the mappings manually by adding/removing additional companies and patterns, rebuild a new training dataset using the [`ipl_dat.py`](./scripts/ipl/ipl_dat.py) script and then retrain the classifiers using [`ipl_train.py`](./scripts/ipl/ipl_train.py).

#### Feature Extraction
The second challenge before training `IPL` refers to the fact that a substantial part of job postings' texts does not contain any information about the employer that published the particular job posting. Instead, it describes, for example, tasks and responsibilities of the position, which can be very similar across industries. Think, for example, about a secratary in the pharmaceutical industry or within a law firm: Both of them have likely similar tasks. These text-snippets do not contain any signal for a industry classification and should thus be excluded to reduce noise in the feature span. 

`IPL` implements a strategy to accomplish this based on a combination of NLP techniques and a pre-trained language model (the procedure is applied to all job postings that were retrieved from <a href="https://github.com/cieb-unibas/jpod">JPOD</a> according to the labelling strategy above).

First, each job posting is tokenized into sentences and every sentence that mentions an employer's name is considered to contain relevant information for classification and extracted from the postings text. Second, a zero-shot classification model is used to extract additional relevant sentences from the remaining text. For this purpose `IPL` builds on the <a href="https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli">`multilingual-MiniLMv2-L6-mnli-xnli` model</a>, a transformer model that is a finetuned version of <a href="https://huggingface.co/xlm-roberta-large">`XLM-RoBERTa-large`</a> for multilingual zero-shot-classification. The model was finetuned on the `XNLI` and `MNLI` datasets and is available on huggingface. `IPL` provides this classifier with the following targets to one of which it assigns each sentence of a job posting: `["who we are", "who this is", "industry or sector", "other"]` (note that you can modify these targets). Job postings' sentences are then sent to this classifier, and if the classifier assigns it to one of the first three tragets, the sentence is extracted and considered to also contain relevant information for industry classifcation. 

The overall proecdure is handeled by the [`IPL.DescExtractor`](./jpap/ipl/DescExtractor.py) class, which is implemented in [this script](./scripts/ipl/ipl_dat.py) and also runs under the hood of the actual [`IPL`](./jpap/ipl/IPL.py) class.

#### Classification
The final step is to train the actual IPL classifier. The approach is based on **transfer learning**, as the `IPL` classifier is a fine-tuned version of the [`XLM-RoBERTa`](https://huggingface.co/docs/transformers/model_doc/xlm-roberta) model, which in itself is a multilingual version of [`RoBERTa`](https://huggingface.co/docs/transformers/model_doc/roberta) that has been pre-trained on 2.5TB of filtered CommonCrawl data containing 100 languages by researchers at [Meta AI](https://ai.meta.com/). **`IPL` is therefore capable to classify job postings independent of their language**.

Before training, an final important issue has to be considered though: Job postings of the same employer can be very similar and if this is nod accounted for properly, the classifier may suffer from **data leakage**. For example, one job posting of Novartis could land in the training set and another one in the testing set even though they are (practically) identical and lead to data leakage. A remedy for this is to use an appropriate sampling procedure when splitting the data into training and test set.

For training IPL, the overall dataset was therefore not split randomly across individual samples but by employers. This ensures that all postings from a certain employer can either be in the test or in the validation set. Furthermore, employer's names in the texts are blinded so that the classifier does not learn to classify postings to industries based on employer names (e.g., it could simply learn that the name 'novartis' refers to the pharmaceutical industry).

The overall training procedure is specified in [this script](./scripts/ipl/ipl_train.py) and resulting performance scores of the finetuned classifiers can be checked in the [`scripts/ipl/logs/`](./scripts/ipl/logs/) directory.

## Jop Postings Translation Pipeline (TPL)
Multilingual translations for job postings.

- Work in Progress
- Consider using a translation pipeline using fine-tuned encoder-decoder architecture: Check the following [paper](https://arxiv.org/pdf/2010.11125.pdf), [code](https://github.com/kadirnar/Multilingual-Translation) and [model](https://huggingface.co/facebook/m2m100_418M) from Meta AI.
