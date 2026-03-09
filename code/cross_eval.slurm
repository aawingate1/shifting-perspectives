#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=cross_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH -e slurms/%j.err
#SBATCH -o slurms/%j.out
#SBATCH --no-requeue

cd /scratch/gpfs/KNORMAN/aw1907/shifting-perspectives/code
source ../venv/bin/activate
mkdir -p slurms

export HF_HOME=/scratch/gpfs/KNORMAN/aw1907/.cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HUB_CACHE}
mkdir -p "${HF_HUB_CACHE}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python 7_cross_evaluation.py mistralai/Mistral-7B-Instruct-v0.1
