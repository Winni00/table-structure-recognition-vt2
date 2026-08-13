#!/bin/bash
#SBATCH --job-name=tf_ptn_hf_72dpi
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
JSONL=/cluster/home/trinhwin/vt2/docling/data/pubtabnet_hf/extracted/pubtabnet/PubTabNet_2.0.0.jsonl
IMG_ROOT=/cluster/home/trinhwin/vt2/docling/data/pubtabnet_hf/extracted/pubtabnet/val_72dpi
VAL_COUNT=9115
FILTER_COUNT=6846

echo "Using PubTabNet HF 72dpi images: $IMG_ROOT"
echo "DPI manifest: $DATA_DIR/val_72dpi_manifest.json"

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
  --output-dir /cluster/home/trinhwin/vt2/docling/results/tableformer_pubtabnet_hf_72dpi/val_full

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
  --output-dir /cluster/home/trinhwin/vt2/docling/results/tableformer_pubtabnet_hf_72dpi/val_size_1x1_20x10
