#!/bin/bash
#SBATCH --job-name=eval_pubtabnet_val_original
#SBATCH --time=3:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --account=cai_exp
#SBATCH --partition=gpu_top_ia
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
PY_TFLOP=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_val_9115_original_gt

$PY_TFLOP $TFLOP_REPO/evaluate_ted.py \
  --model_inference_pathdir $OUT_DIR \
  --output_savepath $OUT_DIR
