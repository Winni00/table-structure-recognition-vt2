#!/bin/bash
#SBATCH --job-name=ftn50k_teds
#SBATCH --time=4:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --array=0-15%4
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
RUN_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn50k_eval
TASK_ID="${SLURM_ARRAY_TASK_ID}"
NUM_SHARDS=16

if [ -f "$RUN_DIR/SKIPPED" ]; then
  echo "50K eval skipped; skipping TEDS shard."
  exit 0
fi

test -f "$RUN_DIR/full_model_inference.json"

$PY scripts/evaluate_ted_shard.py \
  --run-dir "$RUN_DIR" \
  --shard-index "$TASK_ID" \
  --num-shards "$NUM_SHARDS" \
  --num-processes 4 \
  --batch-size 50

$PY scripts/rescore_tflop_fintabnet_canonicalized.py \
  --run-dir "$RUN_DIR" \
  --output-name "ted_score_output_ftn_canonicalized_shard_${TASK_ID}_of_${NUM_SHARDS}.json" \
  --num-processes 4 \
  --batch-size 50 \
  --shard-index "$TASK_ID" \
  --num-shards "$NUM_SHARDS"
