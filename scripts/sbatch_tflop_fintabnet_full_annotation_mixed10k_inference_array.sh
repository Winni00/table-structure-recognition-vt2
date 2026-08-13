#!/bin/bash
#SBATCH --job-name=ftn_mix10k_inf
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
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
PY_SYS=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP

BASE_INPUT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages
BASE_SHARDS=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn100k_eval/inference_shard_inputs
RUN_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_mixed10k_lr1e5_eval

MODEL_DIR=$($PY_SYS scripts/find_latest_mixed_ftn_ptn_checkpoint.py)
EXP_CONFIG=/cluster/home/trinhwin/vt2/docling/results/tflop_mixed_ftn16k_ptn4k_train_10k_lr1e5/tflop_mixed_ftn16k_ptn4k/from_public_checkpoint_10000steps_lr1e5/config.yaml
MODEL_CONFIG=${MODEL_DIR}/config_infer_float16.json
$PY_SYS - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
data["torch_dtype"] = "float16"
(model_dir / "config_infer_float16.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
PY

TASK_ID="${SLURM_ARRAY_TASK_ID}"
SHARD_NAME=$(printf "shard_%02d" "$TASK_ID")
INPUT_DIR="$RUN_DIR/inference_shard_inputs"
SAVE_DIR="$RUN_DIR/inference_shards/$SHARD_NAME"

mkdir -p "$RUN_DIR" "$SAVE_DIR"
ln -sfn "$BASE_INPUT/images" "$RUN_DIR/images"
ln -sfn "$BASE_SHARDS" "$INPUT_DIR"

test -f "${INPUT_DIR}/${SHARD_NAME}_aux.json"
test -f "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"
test -f "$MODEL_CONFIG"
test -f "$EXP_CONFIG"

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
