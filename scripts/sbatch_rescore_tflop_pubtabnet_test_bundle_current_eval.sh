#!/bin/bash
#SBATCH --job-name=rescore_ptn_bundle
#SBATCH --time=6:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs results/tflop_pubtabnet_test_9064_bundle_current_eval

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
SRC_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_test_9064
SAVE_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_test_9064_bundle_current_eval

ln -sf "${SRC_DIR}/full_model_inference.json" "${SAVE_DIR}/full_model_inference.json"

$PY $TFLOP_REPO/evaluate_ted.py \
  --model_inference_pathdir $SAVE_DIR \
  --output_savepath $SAVE_DIR
