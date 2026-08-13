#!/bin/bash
#SBATCH --job-name=purepdf_fix_canon
#SBATCH --time=6:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
RUN_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_updated_crops_broadbestrot_pure_pymupdf_rotfix_public

$PY scripts/rescore_paper_collection_gt_canonicalized.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/gt_canonicalized_rescore" \
  --num-processes 16 \
  --batch-size 50

echo "Pure PyMuPDF rotation-fixed GT-canonicalized rescore complete."
