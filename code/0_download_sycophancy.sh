#!/bin/bash
# Download SycophancyEval datasets from meg-tong/sycophancy-eval
# and Anthropic sycophancy eval datasets

RAW_DIR="$(cd "$(dirname "$0")/.." && pwd)/raw_data/sycophancy"
mkdir -p "$RAW_DIR"
echo "Output directory: $RAW_DIR"

echo "Downloading SycophancyEval datasets from meg-tong/sycophancy-eval..."

# meg-tong datasets (free-form sycophancy evaluation)
wget -q -O "$RAW_DIR/answer.jsonl" \
  "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/answer.jsonl" \
  && echo "  ✓ answer.jsonl" || echo "  ✗ answer.jsonl FAILED"

wget -q -O "$RAW_DIR/are_you_sure.jsonl" \
  "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/are_you_sure.jsonl" \
  && echo "  ✓ are_you_sure.jsonl" || echo "  ✗ are_you_sure.jsonl FAILED"

wget -q -O "$RAW_DIR/feedback.jsonl" \
  "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/feedback.jsonl" \
  && echo "  ✓ feedback.jsonl" || echo "  ✗ feedback.jsonl FAILED"

echo ""
echo "Downloading Anthropic sycophancy eval datasets..."
echo "(These are Git LFS files — cloning the repo to get the actual content)"

# Clone the anthropics/evals repo (sparse checkout for just sycophancy)
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
git clone --filter=blob:none --sparse https://github.com/anthropics/evals.git 2>/dev/null
cd evals
git sparse-checkout set sycophancy 2>/dev/null
# LFS pull to get the actual file contents
git lfs pull --include="sycophancy/*" 2>/dev/null || true

# Copy the sycophancy JSONL files
for f in sycophancy/*.jsonl; do
    fname=$(basename "$f")
    # Check it's a real file (not an LFS pointer)
    if head -1 "$f" | grep -q "^{"; then
        cp "$f" "$RAW_DIR/$fname"
        echo "  ✓ $fname"
    else
        echo "  ⚠ $fname appears to be an LFS pointer, trying HuggingFace fallback..."
        wget -q -O "$RAW_DIR/$fname" \
          "https://huggingface.co/datasets/Anthropic/model-written-evals/resolve/main/sycophancy/$fname" \
          && echo "  ✓ $fname (from HuggingFace)" || echo "  ✗ $fname FAILED"
    fi
done

# Cleanup temp dir
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "All downloads complete! Files saved to $RAW_DIR"
ls -lh "$RAW_DIR"
