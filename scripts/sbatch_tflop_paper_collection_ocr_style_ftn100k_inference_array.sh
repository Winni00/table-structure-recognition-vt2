#!/bin/bash
#SBATCH --job-name=paper_ftn100k_inf
#SBATCH --time=3:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu_ia
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
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_ocr_style_707_ftn100k
TRAIN_ROOT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_full_100k_training/tflop_fintabnet_full_train/from_ftn50k_checkpoint_100000steps_trainonly

TASK_ID="${SLURM_ARRAY_TASK_ID}"
SHARD_NAME=$(printf "shard_%02d" "$TASK_ID")
INPUT_DIR="$OUT_DIR/inference_shard_inputs"
SAVE_DIR="$OUT_DIR/inference_shards/$SHARD_NAME"

mkdir -p "$OUT_DIR" "$SAVE_DIR"
ln -sfn "$SOURCE_DIR/images" "$OUT_DIR/images"
ln -sfn "$SOURCE_DIR/aux.json" "$OUT_DIR/aux.json"
ln -sfn "$SOURCE_DIR/aux_rec.pkl" "$OUT_DIR/aux_rec.pkl"
ln -sfn "$SOURCE_DIR/manifest.json" "$OUT_DIR/manifest.json"
ln -sfn "$SOURCE_DIR/skipped_samples.json" "$OUT_DIR/skipped_samples.json"
ln -sfn "$SOURCE_DIR/post_ocr_validation.json" "$OUT_DIR/post_ocr_validation.json"
ln -sfn "$SOURCE_DIR/inference_shard_inputs" "$INPUT_DIR"

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

echo "Running paper-collection TFLOP inference with FTN-100K checkpoint shard ${TASK_ID}:"
echo "  model=${MODEL_100K}"
echo "  model_config=${MODEL_CONFIG}"
echo "  aux_json=${INPUT_DIR}/${SHARD_NAME}_aux.json"
echo "  aux_rec=${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"
echo "  images=${OUT_DIR}/images"
echo "  save_dir=${SAVE_DIR}"

test -f "${INPUT_DIR}/${SHARD_NAME}_aux.json"
test -f "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl"
test -f "${MODEL_100K}/pytorch_model.bin"
test -f "$MODEL_CONFIG"

$PY $TFLOP_REPO/test.py \
  --tokenizer_name_or_path "$MODEL_100K" \
  --model_name_or_path "$MODEL_100K" \
  --exp_config_path /cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_official_repro/config.yaml \
  --model_config_path "$MODEL_CONFIG" \
  --aux_json_path "${INPUT_DIR}/${SHARD_NAME}_aux.json" \
  --aux_img_path "$OUT_DIR/images" \
  --aux_rec_pkl_path "${INPUT_DIR}/${SHARD_NAME}_aux_rec.pkl" \
  --batch_size 1 \
  --save_dir "$SAVE_DIR" \
  --use_validation
