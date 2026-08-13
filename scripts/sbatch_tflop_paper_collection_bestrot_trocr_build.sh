#!/bin/bash
#SBATCH --job-name=paperbest_trocr
#SBATCH --time=4:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40spcie:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu_top_ia
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=${PY:-/cluster/home/trinhwin/vt2/docling/.venv/bin/python}
MASTER_DIR=${MASTER_DIR:-/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_updated_crops_broadbestrot_master_public}
OUT_DIR=${OUT_DIR:-/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_updated_crops_broadbestrot_trocr_public}

$PY scripts/build_paper_collection_trocr_text_from_run.py \
  --run-dir "$MASTER_DIR" \
  --output-dir "$OUT_DIR" \
  --batch-size 64 \
  --device cuda \
  --max-new-tokens 64 \
  --fallback-to-master

/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python scripts/validate_tflop_ocr_style_aux.py \
  --aux-json "$OUT_DIR/aux.json" \
  --images-dir "$OUT_DIR/images" \
  --aux-rec-pkl "$OUT_DIR/aux_rec.pkl" \
  --require-aux-rec \
  --allow-empty-rec \
  --output-json "$OUT_DIR/post_ocr_validation.json"

rm -rf "$OUT_DIR/inference_shard_inputs"
/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python scripts/split_tflop_aux_bundle.py \
  --aux-json "$OUT_DIR/aux.json" \
  --aux-rec-pkl "$OUT_DIR/aux_rec.pkl" \
  --output-dir "$OUT_DIR/inference_shard_inputs" \
  --num-shards 8 \
  --prefix shard \
  --summary-json "$OUT_DIR/inference_shard_inputs/summary.json"

echo "Built TrOCR aux_rec.pkl and prepared inference shards."
