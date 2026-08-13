#!/bin/bash
#SBATCH --job-name=paper_ftn10kptn_inf
#SBATCH --time=3:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu
#SBATCH --array=0-7%4
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
SOURCE_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_ocr_style_707
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_ocr_style_707_ftn10k_ptnstyle
TRAIN_ROOT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_10k_ptnstyle_training/tflop_fintabnet_train_20k_ptnstyle/from_public_checkpoint_10000steps_trainonly
EXP_CONFIG=${TRAIN_ROOT}/config.yaml

TASK_ID="${SLURM_ARRAY_TASK_ID}"
SHARD_NAME=$(printf "shard_%02d" "$TASK_ID")
INPUT_DIR="$SOURCE_DIR/inference_shard_inputs"
SAVE_DIR="$OUT_DIR/inference_shards/$SHARD_NAME"

mkdir -p "$OUT_DIR" "$SAVE_DIR"
ln -sfn "$SOURCE_DIR/images" "$OUT_DIR/images"
ln -sfn "$SOURCE_DIR/aux.json" "$OUT_DIR/aux.json"
ln -sfn "$SOURCE_DIR/aux_rec.pkl" "$OUT_DIR/aux_rec.pkl"
ln -sfn "$SOURCE_DIR/manifest.json" "$OUT_DIR/manifest.json"
ln -sfn "$SOURCE_DIR/skipped_samples.json" "$OUT_DIR/skipped_samples.json"
ln -sfn "$SOURCE_DIR/post_ocr_validation.json" "$OUT_DIR/post_ocr_validation.json"
ln -sfn "$SOURCE_DIR/inference_shard_inputs" "$OUT_DIR/inference_shard_inputs"

MODEL_DIR=$($PY_SYS - <<'PY'
from pathlib import Path
import re

root = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_10k_ptnstyle_training/tflop_fintabnet_train_20k_ptnstyle/from_public_checkpoint_10000steps_trainonly")
candidates = []
for path in root.glob("epoch_*_step_*"):
    if not (path / "pytorch_model.bin").exists() or not (path / "config.json").exists():
        continue
    match = re.search(r"_step_(\d+)$", path.name)
    step = int(match.group(1)) if match else -1
    candidates.append((step, path))
if not candidates:
    raise SystemExit(f"No checkpoint found under {root}")
print(max(candidates, key=lambda item: item[0])[1])
PY
)

MODEL_CONFIG="${MODEL_DIR}/config_infer_float16.json"
$PY_SYS - "$MODEL_DIR" <<'PY'
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

test -f "${INPUT_DIR}/${SHARD_NAME}_aux.json"
test -f "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"
test -f "${MODEL_DIR}/pytorch_model.bin"
test -f "$MODEL_CONFIG"
test -f "$EXP_CONFIG"

$PY $TFLOP_REPO/test.py \
  --tokenizer_name_or_path "$MODEL_DIR" \
  --model_name_or_path "$MODEL_DIR" \
  --exp_config_path "$EXP_CONFIG" \
  --model_config_path "$MODEL_CONFIG" \
  --aux_json_path "${INPUT_DIR}/${SHARD_NAME}_aux.json" \
  --aux_img_path "$OUT_DIR/images" \
  --aux_rec_pkl_path "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl" \
  --batch_size 1 \
  --save_dir "$SAVE_DIR" \
  --use_validation
