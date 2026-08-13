#!/bin/bash
#SBATCH --job-name=ftn10k_merge_inf
#SBATCH --time=30:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
RUN_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn10k_eval

$PY scripts/merge_tflop_inference_shards.py \
  --shard-dir "$RUN_DIR/inference_shards" \
  --output-json "$RUN_DIR/full_model_inference.json" \
  --summary-json "$RUN_DIR/inference_merge_summary.json"

echo "Merged FTN 10K fine-tuned full annotation inference shards."
