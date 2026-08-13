#!/bin/bash
#SBATCH --job-name=tf_pubtabnet_val_9115
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu_top
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_tableformer_pubtabnet_val_9115.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_tableformer_pubtabnet_val_9115.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling

/cluster/home/trinhwin/vt2/docling/.venv/bin/python -u /cluster/home/trinhwin/vt2/docling/tableformer_pubtabnet_repro.py \
  --split val \
  --subset-size 9115 \
  --device cuda \
  --num-threads 1 \
  --no-viz \
  --html-mode cell_grid \
  --token-mode cell \
  --output-dir /cluster/home/trinhwin/vt2/docling/results/tableformer_pubtabnet_repro/val_full_9115
