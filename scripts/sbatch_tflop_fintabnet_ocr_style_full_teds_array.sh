#!/bin/bash
#SBATCH --job-name=ftn_ocr_teds_arr
#SBATCH --time=4:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --array=0-15%4
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_ocr_style_full_10622

$PY scripts/evaluate_ted_shard.py \
  --run-dir "$OUT_DIR" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards 16 \
  --num-processes 8 \
  --batch-size 50
