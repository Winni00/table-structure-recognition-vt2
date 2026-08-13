#!/usr/bin/env bash
#SBATCH --job-name=target_domain_replay_report
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --time=30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT="/cluster/home/trinhwin/vt2/docling"
cd "$ROOT"

source "$ROOT/.venv-tflop39/bin/activate"
python scripts/report_target_domain_replay_smoke_results.py
