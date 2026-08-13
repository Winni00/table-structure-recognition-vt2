#!/bin/bash
#SBATCH --job-name=mixed_ftn_ptn_build
#SBATCH --time=3:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_mixed_ftn16k_ptn4k_for_10k

$PY scripts/build_tflop_mixed_ftn_ptn_dataset.py \
  --out-dir "$OUT_DIR" \
  --ftn-train-samples 16000 \
  --ptn-train-samples 4000 \
  --ftn-val-samples 800 \
  --ptn-val-samples 200 \
  --bbox-token-cnt 640 \
  --max-length 1376 \
  --seed 42 \
  --force

echo "Built mixed FTN+PTN training dataset in $OUT_DIR"
