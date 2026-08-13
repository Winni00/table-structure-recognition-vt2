#!/bin/bash
#SBATCH --job-name=paperbest_pdfbuild
#SBATCH --time=4:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=${PY:-/cluster/home/trinhwin/vt2/docling/.venv/bin/python}
MASTER_DIR=${MASTER_DIR:-/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_updated_crops_bestrot_master_public}
OUT_DIR=${OUT_DIR:-/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_updated_crops_bestrot_pymupdf_public}
FALLBACK_TO_MASTER=${FALLBACK_TO_MASTER:-1}
if [[ "$FALLBACK_TO_MASTER" == "1" ]]; then
  FALLBACK_ARG=--fallback-to-master
else
  FALLBACK_ARG=--no-fallback-to-master
fi

$PY scripts/build_paper_collection_pymupdf_text_from_run.py \
  --run-dir "$MASTER_DIR" \
  --output-dir "$OUT_DIR" \
  --coord-manifest /cluster/home/trinhwin/vt2/docling/results/paper_collection_updated_crops_coord_check/manifest.json \
  --pad-points 1.5 \
  "$FALLBACK_ARG"

rm -rf "$OUT_DIR/inference_shard_inputs"
/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python scripts/split_tflop_aux_bundle.py \
  --aux-json "$OUT_DIR/aux.json" \
  --aux-rec-pkl "$OUT_DIR/aux_rec.pkl" \
  --output-dir "$OUT_DIR/inference_shard_inputs" \
  --num-shards 8 \
  --prefix shard \
  --summary-json "$OUT_DIR/inference_shard_inputs/summary.json"

echo "Built best-rotation PyMuPDF aux_rec.pkl and prepared inference shards."
