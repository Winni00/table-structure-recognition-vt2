#!/bin/bash
#SBATCH --job-name=ftn_psegt_inf
#SBATCH --time=8:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu
#SBATCH --array=0-15%3
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_pse_matched_annotation_text_full
TASK_ID="${SLURM_ARRAY_TASK_ID}"
SHARD_NAME=$(printf "shard_%02d" "$TASK_ID")
INPUT_DIR="$OUT_DIR/inference_shard_inputs"
SAVE_DIR="$OUT_DIR/inference_shards/$SHARD_NAME"

mkdir -p "$SAVE_DIR"

echo "Running FTN PSE-matched annotation-text public-checkpoint shard ${TASK_ID}:"
echo "  aux_json=${INPUT_DIR}/${SHARD_NAME}_aux.json"
echo "  aux_rec=${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"
echo "  images=${OUT_DIR}/images"
echo "  save_dir=${SAVE_DIR}"

test -f "${INPUT_DIR}/${SHARD_NAME}_aux.json"
test -f "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"
test -d "$OUT_DIR/images"

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
