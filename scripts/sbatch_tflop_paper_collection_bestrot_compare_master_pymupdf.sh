#!/bin/bash
#SBATCH --job-name=paperbest_compare
#SBATCH --time=3:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
MASTER_DIR=${MASTER_DIR:-/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_updated_crops_bestrot_master_public}
PDF_DIR=${PDF_DIR:-/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_updated_crops_bestrot_pymupdf_public}
COMPARE_DIR=${COMPARE_DIR:-/cluster/home/trinhwin/vt2/docling/results/paper_collection_updated_crops_bestrot_master_vs_pymupdf_visuals}

$PY scripts/compare_master_vs_pymupdf_visuals.py \
  --master-run "$MASTER_DIR" \
  --pymupdf-run "$PDF_DIR" \
  --output-dir "$COMPARE_DIR"

echo "Best-rotation MASTER vs PyMuPDF visual comparison complete."
