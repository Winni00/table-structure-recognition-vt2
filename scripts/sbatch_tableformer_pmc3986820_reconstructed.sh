#!/bin/bash
#SBATCH --job-name=tf_ptn_recon_3986820
#SBATCH --time=00:30:00
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
  --subset-size 1 \
  --device cuda \
  --num-threads 1 \
  --no-viz \
  --html-mode cell_grid \
  --token-mode cell \
  --cell-bbox-mode reconstructed \
  --filenames-file /cluster/home/trinhwin/vt2/docling/results/tableformer_pubtabnet_hf_bbox_reconstruction/pmc3986820_one_sample.txt \
  --pubtabnet-jsonl /cluster/home/trinhwin/vt2/docling/data/pubtabnet_hf/extracted/pubtabnet/PubTabNet_2.0.0.jsonl \
  --pubtabnet-images-dir /cluster/home/trinhwin/vt2/docling/data/pubtabnet_hf/extracted/pubtabnet \
  --output-dir /cluster/home/trinhwin/vt2/docling/results/tableformer_pubtabnet_hf_bbox_reconstruction/pmc3986820_fixed_parser_batch
