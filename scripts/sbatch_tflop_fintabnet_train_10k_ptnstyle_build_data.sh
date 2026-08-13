#!/bin/bash
#SBATCH --job-name=ftn10k_ptnstyle_build
#SBATCH --time=04:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY_BUILD=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
DATA_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_20k_ptnstyle_for_10k

$PY_BUILD scripts/build_ftn_trainval_tflop_dataset.py \
  --out-dir "$DATA_DIR" \
  --train-samples 20000 \
  --val-samples 1000 \
  --max-cells 140 \
  --render-scale 2.0 \
  --bbox-token-cnt 640 \
  --max-length 1376 \
  --section-mode first_row_thead \
  --force

echo "Built PTN-style-section FTN train/validation dataset for 10K fine-tune at $DATA_DIR"
