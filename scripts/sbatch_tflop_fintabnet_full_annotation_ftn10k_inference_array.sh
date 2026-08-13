#!/bin/bash
#SBATCH --job-name=ftn10k_full_inf
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
PY_BUILD=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP

BASE_INPUT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages
RUN_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn10k_eval
TRAIN_ROOT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_10k_training/tflop_fintabnet_train_20k/from_public_checkpoint_10000steps_trainonly
EXP_CONFIG=${TRAIN_ROOT}/config.yaml

MODEL_DIR=$($PY_BUILD - <<'PY'
from pathlib import Path
import re

root = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_10k_training/tflop_fintabnet_train_20k/from_public_checkpoint_10000steps_trainonly")
candidates = []
for path in root.glob("epoch_*_step_*"):
    if not (path / "pytorch_model.bin").exists() or not (path / "config.json").exists():
        continue
    match = re.search(r"_step_(\d+)$", path.name)
    step = int(match.group(1)) if match else -1
    candidates.append((step, path))
if not candidates:
    raise SystemExit(f"No TFLOP checkpoint found under {root}")
print(max(candidates, key=lambda item: item[0])[1])
PY
)

MODEL_CONFIG=${MODEL_DIR}/config_infer_float16.json
$PY_BUILD - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
cfg = model_dir / "config.json"
out = model_dir / "config_infer_float16.json"
data = json.loads(cfg.read_text(encoding="utf-8"))
data["torch_dtype"] = "float16"
out.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(out)
PY

TASK_ID="${SLURM_ARRAY_TASK_ID}"
SHARD_NAME=$(printf "shard_%02d" "$TASK_ID")
INPUT_DIR="$RUN_DIR/inference_shard_inputs"
SAVE_DIR="$RUN_DIR/inference_shards/$SHARD_NAME"

mkdir -p "$SAVE_DIR"
ln -sfn "$BASE_INPUT/images" "$RUN_DIR/images"

test -f "${INPUT_DIR}/${SHARD_NAME}_aux.json"
test -f "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"
test -f "$MODEL_CONFIG"

echo "Running FTN full annotation 10K fine-tuned inference shard ${TASK_ID}"
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
