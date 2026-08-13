#!/bin/bash
#SBATCH --job-name=rescore_ptn_orig_eval
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
TFLOP_ORIGINAL=/cluster/home/trinhwin/vt2/docling/repo/TFLOP_clean

BUNDLE_SRC=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_test_9064
RECREATED_SRC=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_test_9064_recreated_ocr

BUNDLE_SAVE=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_test_9064_bundle_original_tflop_eval
RECREATED_SAVE=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_test_9064_recreated_ocr_original_tflop_eval

mkdir -p "${BUNDLE_SAVE}" "${RECREATED_SAVE}"
ln -sf "${BUNDLE_SRC}/full_model_inference.json" "${BUNDLE_SAVE}/full_model_inference.json"
ln -sf "${RECREATED_SRC}/full_model_inference.json" "${RECREATED_SAVE}/full_model_inference.json"

echo "Rescoring bundle OCR with original TFLOP evaluator"
echo "  evaluator=${TFLOP_ORIGINAL}/evaluate_ted.py"
echo "  source=${BUNDLE_SRC}/full_model_inference.json"
echo "  output=${BUNDLE_SAVE}"
$PY "${TFLOP_ORIGINAL}/evaluate_ted.py" \
  --model_inference_pathdir "${BUNDLE_SAVE}" \
  --output_savepath "${BUNDLE_SAVE}"

echo "Rescoring recreated OCR with original TFLOP evaluator"
echo "  evaluator=${TFLOP_ORIGINAL}/evaluate_ted.py"
echo "  source=${RECREATED_SRC}/full_model_inference.json"
echo "  output=${RECREATED_SAVE}"
$PY "${TFLOP_ORIGINAL}/evaluate_ted.py" \
  --model_inference_pathdir "${RECREATED_SAVE}" \
  --output_savepath "${RECREATED_SAVE}"
