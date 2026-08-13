#!/bin/bash
#SBATCH --job-name=ftn_ocr_teds_merge
#SBATCH --time=1:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_ocr_style_full_10622

$PY scripts/merge_ted_shards.py \
  --run-dir "$OUT_DIR" \
  --num-shards 16 \
  --output-name ted_score_output.json

$PY scripts/report_ftn_ocr_style_full_results.py \
  --ocr-run-dir "$OUT_DIR"

echo "Merged sharded TEDS outputs and generated FTN OCR-style report."
