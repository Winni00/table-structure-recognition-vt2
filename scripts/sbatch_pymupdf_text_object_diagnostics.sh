#!/usr/bin/env bash
#SBATCH --job-name=pymupdf_objects
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pymupdf_objects_%j.out
#SBATCH --error=logs/pymupdf_objects_%j.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

rm -rf results/pymupdf_text_object_diagnostics_unrotated

.venv/bin/python scripts/visualize_pymupdf_text_objects.py \
  --run-dir results/tflop_paper_collection_updated_crops_ocr_style_public \
  --pymupdf-run-dir results/tflop_paper_collection_updated_crops_pymupdf_text_public \
  --output-dir results/pymupdf_text_object_diagnostics_unrotated \
  --worst-count 10 \
  --good-count 5 \
  --regions-per-table 6
