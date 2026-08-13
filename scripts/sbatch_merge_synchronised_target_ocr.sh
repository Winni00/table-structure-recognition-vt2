#!/usr/bin/env bash
#SBATCH --job-name=target_domainupd_ocr_merge
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail
ROOT=/cluster/home/trinhwin/vt2/docling
OUT="$ROOT/results/target_domain_train_updated_final_psenet_master"
cd "$ROOT"
.venv/bin/python scripts/merge_tflop_aux_rec_shards.py \
    --shard-dir "$OUT/ocr_shards" \
    --output-pkl "$OUT/aux_rec.pkl" \
    --summary-json "$OUT/aux_rec_merge_summary.json"
.venv/bin/python scripts/validate_tflop_ocr_style_aux.py \
    --aux-json "$OUT/aux.json" \
    --images-dir "$OUT/images" \
    --aux-rec-pkl "$OUT/aux_rec.pkl" \
    --require-aux-rec \
    --allow-empty-rec \
    --output-json "$OUT/post_ocr_validation.json"
touch "$OUT/OCR_COMPLETE"
