#!/bin/bash
#SBATCH --job-name=paperrot_ocr_merge
#SBATCH --time=30:00
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
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_updated_crops_rotation_candidates_public

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

rm -rf "$OUT_DIR/inference_shard_inputs"
$PY scripts/split_tflop_aux_bundle.py \
  --aux-json "$OUT_DIR/aux.json" \
  --aux-rec-pkl "$OUT_DIR/aux_rec.pkl" \
  --output-dir "$OUT_DIR/inference_shard_inputs" \
  --num-shards 3 \
  --prefix shard \
  --summary-json "$OUT_DIR/inference_shard_inputs/summary.json"

echo "Merged rotation-candidate OCR and prepared inference shards."
