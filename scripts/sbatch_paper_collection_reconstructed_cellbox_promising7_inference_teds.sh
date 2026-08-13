#!/bin/bash
#SBATCH --job-name=paper_cellbox7
#SBATCH --time=1:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_reconstructed_cellbox_promising7

echo "Running Paper Collection reconstructed-cellbox mini sanity test:"
echo "  aux_json=${OUT_DIR}/aux.json"
echo "  images=${OUT_DIR}/images"
echo "  aux_rec=${OUT_DIR}/aux_rec.pkl"

test -f "$OUT_DIR/aux.json"
test -f "$OUT_DIR/aux_rec.pkl"
test -d "$OUT_DIR/images"

$PY $TFLOP_REPO/test.py \
  --tokenizer_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --model_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --exp_config_path /cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_official_repro/config.yaml \
  --model_config_path /cluster/home/trinhwin/vt2/docling/models/tflop/config.json \
  --aux_json_path "$OUT_DIR/aux.json" \
  --aux_img_path "$OUT_DIR/images" \
  --aux_rec_pkl_path "$OUT_DIR/aux_rec.pkl" \
  --batch_size 1 \
  --save_dir "$OUT_DIR" \
  --use_validation

$PY $TFLOP_REPO/evaluate_ted.py \
  --model_inference_pathdir "$OUT_DIR" \
  --output_savepath "$OUT_DIR"

/cluster/home/trinhwin/vt2/docling/.venv/bin/python scripts/report_paper_collection_reconstructed_cellbox_mini.py
