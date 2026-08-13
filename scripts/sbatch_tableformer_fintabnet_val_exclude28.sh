#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=tf_fintabnet_val_excl28
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu_top
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_tf_fintabnet_val_excl28.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_tf_fintabnet_val_excl28.err

set -euo pipefail

source /cluster/home/trinhwin/vt2/docling/.venv/bin/activate

python /cluster/home/trinhwin/vt2/docling/tableformer_fintabnet_kaggle_repro.py \
  --table-jsonl /cluster/home/trinhwin/vt2/docling/data/fintabnet_kaggle/FinTabNet_1.0.0_cell_val.jsonl \
  --output-dir /cluster/home/trinhwin/vt2/docling/results/tableformer_fintabnet_kaggle/val_full_excluding_28 \
  --limit 10622 \
  --offset 0 \
  --render-scale 2.0 \
  --device cuda \
  --num-threads 1 \
  --html-mode html_seq_cellmap \
  --token-source cells \
  --only-filenames /cluster/home/trinhwin/vt2/docling/data/fintabnet_val_excluding_28.txt
