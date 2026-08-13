#!/bin/bash
#SBATCH --job-name=ftn_ocr_full_merge
#SBATCH --time=2:00:00
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

PY=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_ocr_style_full_10622

$PY scripts/merge_tflop_aux_rec_shards.py \
  --shard-dir "$OUT_DIR/ocr_shards" \
  --output-pkl "$OUT_DIR/aux_rec.pkl" \
  --summary-json "$OUT_DIR/aux_rec_merge_summary.json"

$PY scripts/validate_tflop_ocr_style_aux.py \
  --aux-json "$OUT_DIR/aux.json" \
  --images-dir "$OUT_DIR/images" \
  --aux-rec-pkl "$OUT_DIR/aux_rec.pkl" \
  --require-aux-rec \
  --allow-empty-rec \
  --output-json "$OUT_DIR/post_ocr_validation.json"

$PY scripts/visualize_tflop_aux_rec_boxes.py \
  --images-dir "$OUT_DIR/images" \
  --aux-rec-pkl "$OUT_DIR/aux_rec.pkl" \
  --output-dir "$OUT_DIR/ocr_visual_debug" \
  --limit 40

rm -rf "$OUT_DIR/inference_shard_inputs"
$PY scripts/split_tflop_aux_bundle.py \
  --aux-json "$OUT_DIR/aux.json" \
  --aux-rec-pkl "$OUT_DIR/aux_rec.pkl" \
  --output-dir "$OUT_DIR/inference_shard_inputs" \
  --num-shards 16 \
  --prefix shard \
  --summary-json "$OUT_DIR/inference_shard_inputs/summary.json"

echo "Merged OCR aux_rec.pkl and prepared TFLOP inference shard inputs."
