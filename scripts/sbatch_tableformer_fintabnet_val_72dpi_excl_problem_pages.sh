#!/bin/bash
#SBATCH --job-name=tf_ftn_72dpi_full
#SBATCH --time=1-00:00:00
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

/cluster/home/trinhwin/vt2/docling/.venv/bin/python \
  /cluster/home/trinhwin/vt2/docling/tableformer_fintabnet_kaggle_repro.py \
  --table-jsonl /cluster/home/trinhwin/vt2/docling/data/fintabnet_kaggle/FinTabNet_1.0.0_cell_val_excluding_28.jsonl \
  --pdf-dir /cluster/home/trinhwin/vt2/docling/data/fintabnet_kaggle/pdfs \
  --limit 10622 \
  --render-scale 1.0 \
  --device cuda \
  --num-threads 1 \
  --html-mode html_seq_cellmap \
  --token-source cells \
  --output-dir /cluster/home/trinhwin/vt2/docling/results/tableformer_fintabnet_kaggle/val_72dpi_excluding_problem_pages
