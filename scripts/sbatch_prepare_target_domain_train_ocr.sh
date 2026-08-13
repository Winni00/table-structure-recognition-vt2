#!/usr/bin/env bash
#SBATCH --job-name=target_domain_ocr_prepare
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail
ROOT=/cluster/home/trinhwin/vt2/docling
OUT="$ROOT/results/target_domain_train_psenet_master"
cd "$ROOT"
mkdir -p logs

test -f results/target_domain_train_upload_status/INVENTORY_COMPLETE
rm -rf "$OUT"
.venv/bin/python scripts/prepare_tflop_paper_collection_ocr_style_inputs.py \
    --raw-root data/target_domain_train \
    --output-dir "$OUT"
.venv/bin/python scripts/split_text_file_shards.py \
    "$OUT/subset.txt" "$OUT/ocr_shards" --num-shards 16 --prefix subset
