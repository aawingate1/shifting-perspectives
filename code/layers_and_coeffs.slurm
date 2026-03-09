#!/bin/bash
#SBATCH --account=knorman
#SBATCH --job-name=syco_vec
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=50G
#SBATCH -e slurms/%j.err
#SBATCH -o slurms/%j.out
#SBATCH --no-requeue

# find the best layers and optimize coefficients for sycophancy

cd /scratch/gpfs/KNORMAN/aw1907/shifting-perspectives/code
source ../venv/bin/activate
mkdir -p slurms

# HuggingFace cache — on /scratch so compute nodes can access it
# PRE-REQUISITE: run this on the login node first to cache the model & dataset, and install matplotlib:
#   source ../venv/bin/activate
#   pip install matplotlib
#   export HF_HOME=/scratch/gpfs/KNORMAN/aw1907/.cache/huggingface
#   python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
#     AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1'); \
#     AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')"
#   python 2b_cache_mmlu.py
export HF_HOME=/scratch/gpfs/KNORMAN/aw1907/.cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HUB_CACHE}
mkdir -p "${HF_HUB_CACHE}"

# Compute nodes have no internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# First find the best layers
# python 4b_optimize_layers_sycophancy.py mistralai/Mistral-7B-Instruct-v0.1

# Then optimize the coefficient for that layer and evaluate MMLU tradeoff
python 5b_optimize_coeff_sycophancy.py mistralai/Mistral-7B-Instruct-v0.1
