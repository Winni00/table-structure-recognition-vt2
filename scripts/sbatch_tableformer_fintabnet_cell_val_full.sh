#!/bin/bash
#SBATCH --job-name=tf_fintabnet_full
#SBATCH --time=24:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu_top
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

/cluster/home/trinhwin/vt2/docling/.venv/bin/python -u \
  /cluster/home/trinhwin/vt2/docling/tableformer_fintabnet_kaggle_repro.py \
  --table-jsonl /cluster/home/trinhwin/vt2/docling/data/fintabnet_kaggle/FinTabNet_1.0.0_cell_val.jsonl \
  --limit 10656 \
  --render-scale 2.0 \
  --device cuda \
  --num-threads 1 \
  --html-mode html_seq_cellmap \
  --output-dir /cluster/home/trinhwin/vt2/docling/results/tableformer_fintabnet_kaggle/cell_val_full_10656_html_seq_cellmap
