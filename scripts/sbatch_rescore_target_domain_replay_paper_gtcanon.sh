#!/usr/bin/env bash
#SBATCH --job-name=target_domain_paper_gtcanon
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --array=0-2%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

ROOT="/cluster/home/trinhwin/vt2/docling"
cd "$ROOT"

source "$ROOT/.venv-tflop39/bin/activate"

variants=(target_domain_only target_domain75_ptn25 target_domain50_ptn50)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"

python scripts/rescore_paper_collection_gt_canonicalized.py \
  --run-dir "results/tflop_target_domain_replay_smoke_1k_eval/${variant}/paper_test" \
  --output-dir "results/tflop_target_domain_replay_smoke_1k_eval/${variant}/paper_test/gt_canonicalized_rescore" \
  --num-processes 12 \
  --batch-size 50
