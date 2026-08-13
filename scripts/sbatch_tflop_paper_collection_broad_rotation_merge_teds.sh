#!/bin/bash
#SBATCH --job-name=paperbrot_teds
#SBATCH --time=1:00:00
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
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_updated_crops_rotation_broad_badcases_public

$PY scripts/merge_tflop_inference_shards.py \
  --shard-dir "$OUT_DIR/inference_shards" \
  --output-json "$OUT_DIR/full_model_inference.json" \
  --summary-json "$OUT_DIR/inference_merge_summary.json"

for SHARD in $(seq 0 5); do
  $PY scripts/evaluate_ted_shard.py \
    --run-dir "$OUT_DIR" \
    --shard-index "$SHARD" \
    --num-shards 6 \
    --num-processes 8 \
    --batch-size 25
done

$PY scripts/merge_ted_shards.py \
  --run-dir "$OUT_DIR" \
  --num-shards 6 \
  --output-name ted_score_output.json

$PY scripts/rescore_paper_collection_gt_canonicalized.py \
  --run-dir "$OUT_DIR" \
  --output-dir "$OUT_DIR/gt_canonicalized_rescore"

$PY scripts/summarize_paper_collection_rotation_experiment.py \
  --run-dir "$OUT_DIR" \
  --scores-json "$OUT_DIR/gt_canonicalized_rescore/ted_score_output.json" \
  --output-name rotation_summary_gt_canon.json \
  --readme-name README_gt_canon.md

$PY scripts/summarize_paper_collection_rotation_experiment.py \
  --run-dir "$OUT_DIR" \
  --output-name rotation_summary_official.json \
  --readme-name README_official.md

echo "Broad-rotation inference and TEDS summary complete."
