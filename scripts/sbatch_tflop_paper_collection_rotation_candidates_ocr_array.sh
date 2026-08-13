#!/bin/bash
#SBATCH --job-name=paperrot_ocr
#SBATCH --time=2:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu
#SBATCH --array=0-2%3
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY_OCR=/cluster/home/trinhwin/vt2/docling/.venv-tablemaster38/bin/python
REPRO=/cluster/home/trinhwin/vt2/docling/repro/pubtabnet_tflop_reproduction
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_updated_crops_rotation_candidates_public
SHARD_DIR="$OUT_DIR/ocr_shards"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
SHARD_SUBSET=$(printf "%s/subset_%02d.txt" "$SHARD_DIR" "$TASK_ID")
SHARD_OUT=$(printf "%s/aux_rec_%02d.pkl" "$SHARD_DIR" "$TASK_ID")

echo "Running rotation-candidate OCR shard ${TASK_ID}:"
echo "  subset=${SHARD_SUBSET}"
echo "  images=${OUT_DIR}/images"
echo "  output=${SHARD_OUT}"

test -f "$SHARD_SUBSET"

$PY_OCR "$REPRO/scripts/run_tablemaster_end2end_subset.py" \
  --subset "$SHARD_SUBSET" \
  --images-dir "$OUT_DIR/images" \
  --output-pkl "$SHARD_OUT" \
  --on-error empty
