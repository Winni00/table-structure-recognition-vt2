#!/bin/bash
#SBATCH --job-name=paper_ftn100k_cfgsmoke
#SBATCH --time=1:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
PY_SYS=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
SOURCE_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_ocr_style_707
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_ocr_style_707_ftn100k_correct_config_smoke
TRAIN_ROOT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_full_100k_training/tflop_fintabnet_full_train/from_ftn50k_checkpoint_100000steps_trainonly
EXP_CONFIG=${TRAIN_ROOT}/config.yaml

SHARD_NAME=shard_00
INPUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_ocr_style_707/inference_shard_inputs
SAVE_DIR=${OUT_DIR}/inference_shards/${SHARD_NAME}

mkdir -p "$OUT_DIR" "$SAVE_DIR"
ln -sfn "$SOURCE_DIR/images" "$OUT_DIR/images"

MODEL_100K=$($PY_SYS - <<'PY'
from pathlib import Path
import re

root = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_full_100k_training/tflop_fintabnet_full_train/from_ftn50k_checkpoint_100000steps_trainonly")
candidates = []
for path in root.glob("epoch_*_step_*"):
    if not (path / "pytorch_model.bin").exists() or not (path / "config.json").exists():
        continue
    match = re.search(r"_step_(\d+)$", path.name)
    step = int(match.group(1)) if match else -1
    candidates.append((step, path))
if not candidates:
    raise SystemExit(f"No 100K checkpoint found under {root}")
print(max(candidates, key=lambda item: item[0])[1])
PY
)

MODEL_CONFIG="${MODEL_100K}/config_infer_float16.json"
$PY_SYS - "$MODEL_100K" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
cfg["torch_dtype"] = "float16"
(model_dir / "config_infer_float16.json").write_text(
    json.dumps(cfg, indent=2), encoding="utf-8"
)
PY

echo "Smoke inference with FTN-100K checkpoint and FTN exp_config"
echo "  model=${MODEL_100K}"
echo "  exp_config=${EXP_CONFIG}"
echo "  aux_json=${INPUT_DIR}/${SHARD_NAME}_aux.json"
echo "  aux_rec=${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"
echo "  save_dir=${SAVE_DIR}"

test -f "${INPUT_DIR}/${SHARD_NAME}_aux.json"
test -f "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"
test -f "${MODEL_100K}/pytorch_model.bin"
test -f "$MODEL_CONFIG"
test -f "$EXP_CONFIG"

$PY $TFLOP_REPO/test.py \
  --tokenizer_name_or_path "$MODEL_100K" \
  --model_name_or_path "$MODEL_100K" \
  --exp_config_path "$EXP_CONFIG" \
  --model_config_path "$MODEL_CONFIG" \
  --aux_json_path "${INPUT_DIR}/${SHARD_NAME}_aux.json" \
  --aux_img_path "$OUT_DIR/images" \
  --aux_rec_pkl_path "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl" \
  --batch_size 1 \
  --save_dir "$SAVE_DIR" \
  --use_validation
