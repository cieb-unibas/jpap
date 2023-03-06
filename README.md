# jpap
jpap ('job postings analyses pipelines') contains different pipelines for in-depth analysis of job postings.

## To-Do's
- pipeline for feature extraction (relevant text snippets)
- choose a pre-trained encoder-architecture
- construct dataset for fine-tuning
- translation pipeline using fine-tuned encoder-decoder architecture
- ...

## Installation

#### Step 1
`cd` into this repository, setup a virtual environment via conda and install dependencies:
```bash
conda create --name jpap python=3.9
conda install -r requirements.txt
```
#### Step 2
Check for GPU support on your machine and setup `PyTorch` according to the [developer website](https://pytorch.org/).

#### Step 3
Install the `transformers` library as specified by [Huggingface](https://huggingface.co/docs/transformers/installation):

```bash
conda install -c huggingface transformers
```

## Job Postings Industry Pipeline (IPL)
Find efficient ways to assign postings to industries.

## Jop Postings Translation Pipeline (TPL)
Multilingual translations for job postings.