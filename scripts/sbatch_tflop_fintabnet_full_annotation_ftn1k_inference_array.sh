#!/bin/bash
#SBATCH --job-name=ftn1k_full_inf
#SBATCH --time=4:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu
#SBATCH --array=0-3%4
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP

BASE_INPUT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages
RUN_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn1k_eval
TRAIN_ROOT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_smoke_1k_training/tflop_fintabnet_train_smoke_1k/from_public_checkpoint_1000steps
MODEL_DIR=${TRAIN_ROOT}/epoch_1_step_1000
EXP_CONFIG=${TRAIN_ROOT}/config.yaml
MODEL_CONFIG=${MODEL_DIR}/config_infer_float16.json

TASK_ID="${SLURM_ARRAY_TASK_ID}"
SHARD_NAME=$(printf "shard_%02d" "$TASK_ID")
INPUT_DIR="$RUN_DIR/inference_shard_inputs"
SAVE_DIR="$RUN_DIR/inference_shards/$SHARD_NAME"

mkdir -p "$SAVE_DIR"
ln -sfn "$BASE_INPUT/images" "$RUN_DIR/images"

test -f "${INPUT_DIR}/${SHARD_NAME}_aux.json"
test -f "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"
test -f "$MODEL_CONFIG"

echo "Running FTN full annotation fine-tuned inference shard ${TASK_ID}"
echo "  model=${MODEL_DIR}"
echo "  aux_json=${INPUT_DIR}/${SHARD_NAME}_aux.json"
echo "  aux_rec=${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"
echo "  images=${RUN_DIR}/images"
echo "  save_dir=${SAVE_DIR}"

$PY $TFLOP_REPO/test.py \
  --tokenizer_name_or_path "$MODEL_DIR" \
  --model_name_or_path "$MODEL_DIR" \
  --exp_config_path "$EXP_CONFIG" \
  --model_config_path "$MODEL_CONFIG" \
  --aux_json_path "${INPUT_DIR}/${SHARD_NAME}_aux.json" \
  --aux_img_path "$RUN_DIR/images" \
  --aux_rec_pkl_path "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl" \
  --batch_size 1 \
  --save_dir "$SAVE_DIR" \
  --use_validation
