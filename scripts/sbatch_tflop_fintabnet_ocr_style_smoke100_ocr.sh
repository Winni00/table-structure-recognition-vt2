#!/bin/bash
#SBATCH --job-name=ftn_ocr_style_100
#SBATCH --time=8:00:00
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
mkdir -p logs

PY_BUILD=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
PY_OCR=/cluster/home/trinhwin/vt2/docling/.venv-tablemaster38/bin/python
REPRO=/cluster/home/trinhwin/vt2/docling/repro/pubtabnet_tflop_reproduction
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_ocr_style_smoke100

echo "Preparing FTN OCR-style smoke inputs:"
echo "  output=${OUT_DIR}"

$PY_BUILD scripts/prepare_tflop_fintabnet_ocr_style_inputs.py \
  --source-dir /cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages \
  --output-dir "$OUT_DIR" \
  --limit 100 \
  --offset 0

$PY_BUILD scripts/validate_tflop_ocr_style_aux.py \
  --aux-json "$OUT_DIR/aux.json" \
  --images-dir "$OUT_DIR/images" \
  --output-json "$OUT_DIR/pre_ocr_validation.json"

echo "Running TableMASTER PSENet+MASTER OCR only:"
echo "  subset=${OUT_DIR}/subset.txt"
echo "  images=${OUT_DIR}/images"
echo "  output=${OUT_DIR}/aux_rec.pkl"

$PY_OCR "$REPRO/scripts/run_tablemaster_end2end_subset.py" \
  --subset "$OUT_DIR/subset.txt" \
  --images-dir "$OUT_DIR/images" \
  --output-pkl "$OUT_DIR/aux_rec.pkl"

$PY_BUILD scripts/validate_tflop_ocr_style_aux.py \
  --aux-json "$OUT_DIR/aux.json" \
  --images-dir "$OUT_DIR/images" \
  --aux-rec-pkl "$OUT_DIR/aux_rec.pkl" \
  --require-aux-rec \
  --output-json "$OUT_DIR/post_ocr_validation.json"

$PY_BUILD scripts/visualize_tflop_aux_rec_boxes.py \
  --images-dir "$OUT_DIR/images" \
  --aux-rec-pkl "$OUT_DIR/aux_rec.pkl" \
  --output-dir "$OUT_DIR/ocr_visual_debug" \
  --limit 20

echo "OCR-style aux_rec.pkl is ready. TFLOP inference is intentionally not started by this job."
