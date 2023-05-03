#!/bin/bash

#SBATCH --job-name=jpap-ipl-train
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=00:30:00
#SBATCH --qos=30mins

#SBATCH --output=jpap-ipl-train
#SBATCH --error=jpap-ipl-train_errors
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=matthias.niggli@unibas.ch
#SBATCH --partition=a100
#SBATCH --gres=gpu:1   


## CUDA 11.6 for pytorch 1.13.1-----
ml load CUDA/....

## cuDNN 8.3.2.44 for pytorch 1.13.1-----
ml load cuDNN/...

## set directory and activate virtual environment ----
cd "..."
source jpap_venv/bin/activate

## run .py script -----
python ./jpap/scripts/ipl/ipl_train.py