"""
Preprocess Anthropic SycophancyEval datasets into contrastive pairs for steering
vector training, and structured validation/test data for evaluation.

Axis: sycophancy — "keeping the user the most pleased" (negative/sycophantic)
                    vs "expressing oneself authentically" (positive/authentic)

Source: Anthropic evals (multiple-choice with biographies)
  - sycophancy_on_nlp_survey.jsonl
  - sycophancy_on_philpapers2020.jsonl
  - sycophancy_on_political_typology_quiz.jsonl

Outputs:
  Train:    contrastive pairs JSON [{"positive": "...", "negative": "..."}, ...]
  Validate: structured JSON  [{"question": "...", "syco_answer": "...",
                                "auth_answer": "...", "positive": "...", "negative": "..."}, ...]
  Test:     same structured JSON format

Run download script first:
  bash 0_download_sycophancy.sh
"""

import os
import json
import random

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
RAW_DIR = "../raw_data/sycophancy"
TRAIN_DIR = "../data/sycophancy_train"
VALIDATE_DIR = "../data/sycophancy_validate"
TEST_DIR = "../data/sycophancy_test"

for d in [TRAIN_DIR, VALIDATE_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helper: load JSONL
# ──────────────────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic multiple-choice sycophancy datasets
#   Format: {"question": "...", "answer_matching_behavior": " (A)",
#            "answer_not_matching_behavior": " (B)"}
#
#   positive = authentic answer (answer_not_matching_behavior)
#   negative = sycophantic answer (answer_matching_behavior)
# ──────────────────────────────────────────────────────────────────────────────
def process_anthropic_mc(filepath):
    """Convert Anthropic MC sycophancy dataset to structured entries."""
    data = load_jsonl(filepath)
    entries = []

    for row in data:
        question = row["question"].strip()
        syco_answer = row["answer_matching_behavior"].strip()       # e.g. "(A)"
        auth_answer = row["answer_not_matching_behavior"]

        # answer_not_matching_behavior can be a list for philpapers
        if isinstance(auth_answer, list):
            auth_answer = auth_answer[0].strip()
        else:
            auth_answer = auth_answer.strip()

        # Build contrastive pair: question text + chosen answer
        positive = f"{question} {auth_answer}"
        negative = f"{question} {syco_answer}"

        entries.append({
            "question": question,
            "syco_answer": syco_answer,
            "auth_answer": auth_answer,
            "positive": positive,
            "negative": negative,
        })

    return entries


# ──────────────────────────────────────────────────────────────────────────────
# Main processing pipeline
# ──────────────────────────────────────────────────────────────────────────────
def main():
    all_entries = []

    anthropic_files = [
        "sycophancy_on_nlp_survey.jsonl",
        "sycophancy_on_philpapers2020.jsonl",
        "sycophancy_on_political_typology_quiz.jsonl",
    ]

    for fname in anthropic_files:
        fpath = os.path.join(RAW_DIR, fname)
        if os.path.exists(fpath):
            entries = process_anthropic_mc(fpath)
            print(f"  {fname}: {len(entries)} contrastive pairs")
            all_entries.extend(entries)
        else:
            print(f"  {fname}: NOT FOUND (skipping)")

    print(f"\nTotal contrastive pairs: {len(all_entries)}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_entries)

    # Split: 300 train, 300 validate, rest test (matching BBQ proportions)
    n_train = 300
    n_validate = 300

    train_entries = all_entries[:n_train]
    validate_entries = all_entries[n_train:n_train + n_validate]
    test_entries = all_entries[n_train + n_validate:]

    # Save train set (contrastive pairs only — for load_and_tokenize_contrastive)
    train_pairs = [{"positive": e["positive"], "negative": e["negative"]}
                   for e in train_entries]
    train_path = os.path.join(TRAIN_DIR, "sycophancy_train.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved train ({len(train_pairs)} pairs) → {train_path}")

    # Save validate set (structured — for evaluation scripts)
    validate_path = os.path.join(VALIDATE_DIR, "sycophancy_validate.json")
    with open(validate_path, "w", encoding="utf-8") as f:
        json.dump(validate_entries, f, indent=2, ensure_ascii=False)
    print(f"Saved validate ({len(validate_entries)} entries) → {validate_path}")

    # Save test set (structured — for evaluation scripts)
    test_path = os.path.join(TEST_DIR, "sycophancy_test.json")
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_entries, f, indent=2, ensure_ascii=False)
    print(f"Saved test ({len(test_entries)} entries) → {test_path}")

    print("\nDone! Sycophancy preprocessing complete.")


if __name__ == "__main__":
    main()
