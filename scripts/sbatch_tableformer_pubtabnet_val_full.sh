#!/bin/bash
#SBATCH --job-name=tf_pubtabnet_val
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
  /cluster/home/trinhwin/vt2/docling/tableformer_pubtabnet_repro.py \
  --split val \
  --subset-size 9115 \
  --device cuda \
  --num-threads 1 \
  --no-viz \
  --html-mode html_seq_cellmap \
  --token-mode cell \
  --output-dir /cluster/home/trinhwin/vt2/docling/results/tableformer_pubtabnet_repro/val_full_9115_html_seq_cellmap
