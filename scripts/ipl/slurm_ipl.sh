#!/bin/bash

#SBATCH --job-name=jpap-ipl-train
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=00:30:00
#SBATCH --qos=30min

#SBATCH --output=scripts/ipl/logs/jpap-ipl-train-unique-employers-macro
#SBATCH --error=scripts/ipl/logs/jpap-ipl-unique-employers-macro_errors
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=matthias.niggli@unibas.ch

#SBATCH --partition=a100
#SBATCH --gres=gpu:1   

## CUDA 11.6 for pytorch 1.13.1-----
ml load CUDA/11.7.0

## set directory and activate virtual environment ----
cd "/scicore/home/weder/GROUP/Innovation/05_job_adds_data/jpap/"
source ../jpap-venv/bin/activate

## run .py script -----
python scripts/ipl/ipl_train.py