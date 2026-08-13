#!/usr/bin/env bash
#SBATCH --job-name=manualrot_pdfdiag
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail
ROOT=/cluster/home/trinhwin/vt2/docling
MASTER="$ROOT/results/tflop_paper_collection_updated_crops_manualrot_master_public"
PDF="$ROOT/results/tflop_paper_collection_updated_crops_manualrot_pure_pymupdf_public"
OUT="$ROOT/results/pymupdf_text_object_diagnostics_manualrot_correct"
cd "$ROOT"

for RUN in "$MASTER" "$PDF"; do
  .venv-tflop39/bin/python scripts/rescore_paper_collection_gt_canonicalized.py \
    --run-dir "$RUN" \
    --output-dir "$RUN/gt_canonicalized_rescore" \
    --num-processes 12 \
    --batch-size 50
done

rm -rf "$OUT"
.venv/bin/python scripts/visualize_pymupdf_text_objects.py \
  --run-dir "$MASTER" \
  --pymupdf-run-dir "$PDF" \
  --output-dir "$OUT" \
  --worst-count 10 \
  --good-count 5 \
  --regions-per-table 6

touch "$OUT/COMPLETE"
