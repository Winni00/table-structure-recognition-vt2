#!/bin/bash
#SBATCH --job-name=ftn_ocr_full_inf
#SBATCH --time=8:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu
#SBATCH --array=0-15%8
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_ocr_style_full_10622
TASK_ID="${SLURM_ARRAY_TASK_ID}"
SHARD_NAME=$(printf "shard_%02d" "$TASK_ID")
INPUT_DIR="$OUT_DIR/inference_shard_inputs"
SAVE_DIR="$OUT_DIR/inference_shards/$SHARD_NAME"

mkdir -p "$SAVE_DIR"

echo "Running TFLOP inference shard ${TASK_ID}:"
echo "  aux_json=${INPUT_DIR}/${SHARD_NAME}_aux.json"
echo "  aux_rec=${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"
echo "  images=${OUT_DIR}/images"
echo "  save_dir=${SAVE_DIR}"

test -f "${INPUT_DIR}/${SHARD_NAME}_aux.json"
test -f "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"

$PY $TFLOP_REPO/test.py \
  --tokenizer_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --model_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --exp_config_path /cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_official_repro/config.yaml \
  --model_config_path /cluster/home/trinhwin/vt2/docling/models/tflop/config.json \
  --aux_json_path "${INPUT_DIR}/${SHARD_NAME}_aux.json" \
  --aux_img_path "$OUT_DIR/images" \
  --aux_rec_pkl_path "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl" \
  --batch_size 1 \
  --save_dir "$SAVE_DIR" \
  --use_validation
