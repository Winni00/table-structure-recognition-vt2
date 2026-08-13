#!/bin/bash
#SBATCH --job-name=tf_ptn_hf_orig
#SBATCH --time=1-00:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu_top
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling

PY=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
DATA_DIR=/cluster/home/trinhwin/vt2/docling/data/pubtabnet_hf

$PY /cluster/home/trinhwin/vt2/docling/prepare_pubtabnet_hf.py \
  --output-dir $DATA_DIR

JSONL=$($PY -c "import json; print(json.load(open('$DATA_DIR/manifest.json'))['jsonl_path'])")
IMG_ROOT=$($PY -c "import json; print(json.load(open('$DATA_DIR/manifest.json'))['image_root'])")
VAL_COUNT=$($PY -c "import json; print(json.load(open('$DATA_DIR/manifest.json'))['val_count'])")
FILTER_COUNT=$($PY -c "import json; print(json.load(open('$DATA_DIR/manifest.json'))['val_1x1_20x10_count'])")

echo "PubTabNet HF JSONL: $JSONL"
echo "PubTabNet HF image root: $IMG_ROOT"
echo "PubTabNet HF val count: $VAL_COUNT"
echo "PubTabNet HF 1x1..20x10 count: $FILTER_COUNT"

$PY /cluster/home/trinhwin/vt2/docling/tableformer_pubtabnet_repro.py \
  --split val \
  --subset-size "$VAL_COUNT" \
  --device cuda \
  --num-threads 1 \
  --no-viz \
  --html-mode html_seq_cellmap \
  --token-mode cell \
  --pubtabnet-jsonl "$JSONL" \
  --pubtabnet-images-dir "$IMG_ROOT" \
  --output-dir /cluster/home/trinhwin/vt2/docling/results/tableformer_pubtabnet_hf_original/val_full

$PY /cluster/home/trinhwin/vt2/docling/tableformer_pubtabnet_repro.py \
  --split val \
  --subset-size "$FILTER_COUNT" \
  --device cuda \
  --num-threads 1 \
  --no-viz \
  --html-mode html_seq_cellmap \
  --token-mode cell \
  --min-rows 1 \
  --max-rows 20 \
  --min-cols 1 \
  --max-cols 10 \
  --pubtabnet-jsonl "$JSONL" \
  --pubtabnet-images-dir "$IMG_ROOT" \
  --output-dir /cluster/home/trinhwin/vt2/docling/results/tableformer_pubtabnet_hf_original/val_size_1x1_20x10
