#!/bin/bash
#SBATCH --job-name=teds_shard
#SBATCH --time=06:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling

: "${RUN_DIR:?Set RUN_DIR to the TFLOP result directory}"
: "${NUM_SHARDS:?Set NUM_SHARDS to the Slurm array size}"

/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python \
  /cluster/home/trinhwin/vt2/docling/scripts/evaluate_ted_shard.py \
  --run-dir "$RUN_DIR" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS" \
  --num-processes "${NUM_PROCESSES:-4}" \
  --batch-size "${BATCH_SIZE:-50}"
