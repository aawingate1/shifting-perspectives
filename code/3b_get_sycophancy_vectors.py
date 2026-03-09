"""
Generate steering vectors for the sycophancy axis.

Mirrors the approach in 3_get_vectors.py:
  1) train:        contrastive pairs from Anthropic sycophancy MC data
  2) train+prompt: same pairs with a sycophancy-aware prompt prefix
  3) generate_ss:  dialz-generated dataset using sentence-starters
  4) generate_qa:  dialz-generated dataset using question-answer format

Usage:
  python 3b_get_sycophancy_vectors.py <model_name>

  e.g.:
  python 3b_get_sycophancy_vectors.py mistralai/Mistral-7B-Instruct-v0.1
"""

import os
import sys
import datetime
from dialz import Dataset, SteeringModel, SteeringVector
from utils import sycophancy_axes, load_and_tokenize_contrastive, contrastive_pairs

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    raise ValueError("Model name must be provided as a command-line argument.")

# Map model names to short names
model_short_names = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen",
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
}

model_short_name = model_short_names.get(model_name)
if not model_short_name:
    raise ValueError(f"Unknown model name: {model_name}")

dirs = {
    "train": f"../vectors/{model_short_name}/train",
    "train+prompt": f"../vectors/{model_short_name}/train+prompt",
    "generate_ss": f"../vectors/{model_short_name}/generate_ss",
    "generate_qa": f"../vectors/{model_short_name}/generate_qa",
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

model = SteeringModel(model_name, [5])  # layer arg is arbitrary; we're just training vectors

for axis in sycophancy_axes:

    print(f"Creating 4 vectors for {axis} at:", datetime.datetime.now())

    ## 1) Dataset contrastive pairs (from Anthropic sycophancy MC data)
    path = f"../data/sycophancy_train/{axis}_train.json"
    train_dataset = load_and_tokenize_contrastive(model_name, path)
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['train'], f"{axis}.gguf"))
    print(f"  ✓ {axis} train vector saved")

    ## 2) Same data + prompt prefix
    train_dataset = load_and_tokenize_contrastive(
        model_name, path,
        prompt="Consider the sycophancy in the following. "
    )
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['train+prompt'], f"{axis}.gguf"))
    print(f"  ✓ {axis} train+prompt vector saved")

    ## 3) Generated Dataset (sentence-starters)
    train_dataset = Dataset.create_dataset(
        model_name,
        contrastive_pairs[axis],
        system_role=" ",
        prompt_type="sentence-starters"
    )
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['generate_ss'], f"{axis}.gguf"))
    print(f"  ✓ {axis} generate_ss vector saved")

    ## 4) Generated Dataset (question-answer)
    train_dataset = Dataset.create_dataset(
        model_name,
        contrastive_pairs[axis],
        system_role=" ",
        prompt_type="question-answer"
    )
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['generate_qa'], f"{axis}.gguf"))
    print(f"  ✓ {axis} generate_qa vector saved")

print(f"\nAll sycophancy vectors done at:", datetime.datetime.now())
