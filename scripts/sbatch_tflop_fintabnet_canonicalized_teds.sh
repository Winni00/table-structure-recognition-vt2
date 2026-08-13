#!/bin/bash
#SBATCH --job-name=ftn_teds_canon
#SBATCH --time=08:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling

/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python \
  /cluster/home/trinhwin/vt2/docling/scripts/rescore_tflop_fintabnet_canonicalized.py \
  --run-dir /cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages \
  --num-processes 8 \
  --batch-size 200
