#!/bin/bash
#SBATCH --job-name=dl_fintabnet_cell_val
#SBATCH --time=12:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --account=cai_exp
#SBATCH --partition=gpu_top
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

/cluster/home/trinhwin/vt2/docling/.venv/bin/python -u \
  /cluster/home/trinhwin/vt2/docling/download_fintabnet_cell_val_pdfs.py
