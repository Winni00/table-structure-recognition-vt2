#!/bin/bash
#SBATCH --job-name=ftn_canon_shard
#SBATCH --time=04:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling

: "${NUM_SHARDS:?Set NUM_SHARDS to the Slurm array size}"

/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python \
  /cluster/home/trinhwin/vt2/docling/scripts/rescore_tflop_fintabnet_canonicalized.py \
  --run-dir /cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages \
  --output-name "ted_score_output_ftn_canonicalized_shard_${SLURM_ARRAY_TASK_ID}_of_${NUM_SHARDS}.json" \
  --num-processes "${NUM_PROCESSES:-4}" \
  --batch-size "${BATCH_SIZE:-50}" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS"
