#!/bin/bash
#SBATCH --job-name=tf_ptn_val_png_full
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
  /cluster/home/trinhwin/vt2/docling/tableformer_pubtabnet_repro.py \
  --split val \
  --subset-size 9115 \
  --device cuda \
  --num-threads 1 \
  --no-viz \
  --html-mode html_seq_cellmap \
  --token-mode cell \
  --output-dir /cluster/home/trinhwin/vt2/docling/results/tableformer_pubtabnet_repro/val_9115_existingpng_72dpi_reference
