#!/bin/bash
#SBATCH --job-name=paperbest_teds
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
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
OUT_DIR=${OUT_DIR:-/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_updated_crops_bestrot_master_public}

$PY scripts/merge_tflop_inference_shards.py \
  --shard-dir "$OUT_DIR/inference_shards" \
  --output-json "$OUT_DIR/full_model_inference.json" \
  --summary-json "$OUT_DIR/inference_merge_summary.json"

for SHARD in $(seq 0 7); do
  $PY scripts/evaluate_ted_shard.py \
    --run-dir "$OUT_DIR" \
    --shard-index "$SHARD" \
    --num-shards 8 \
    --num-processes 8 \
    --batch-size 50
done

$PY scripts/merge_ted_shards.py \
  --run-dir "$OUT_DIR" \
  --num-shards 8 \
  --output-name ted_score_output.json

$PY scripts/report_paper_collection_ocr_style_results.py \
  --run-dir "$OUT_DIR" \
  --output-dir "$OUT_DIR/report" \
  --examples-per-group 5

$PY scripts/rescore_paper_collection_gt_canonicalized.py \
  --run-dir "$OUT_DIR" \
  --output-dir "$OUT_DIR/gt_canonicalized_rescore"

echo "Best-rotation MASTER inference and TEDS complete."
