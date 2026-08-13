#!/bin/bash
#SBATCH --job-name=ftn_ocr_full_eval
#SBATCH --time=8:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_ocr_style_full_10622

$PY scripts/merge_tflop_inference_shards.py \
  --shard-dir "$OUT_DIR/inference_shards" \
  --output-json "$OUT_DIR/full_model_inference.json" \
  --summary-json "$OUT_DIR/inference_merge_summary.json"

$PY $TFLOP_REPO/evaluate_ted.py \
  --model_inference_pathdir "$OUT_DIR" \
  --output_savepath "$OUT_DIR"

echo "Full FTN OCR-style TFLOP inference and official TEDS evaluation complete."
