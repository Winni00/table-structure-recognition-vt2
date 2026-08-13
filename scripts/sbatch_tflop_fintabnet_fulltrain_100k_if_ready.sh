#!/bin/bash
#SBATCH --job-name=tflop_ftn_full100k
#SBATCH --time=3-00:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
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

PY=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
PY_TFLOP=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
DATA_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_full_for_longrun
STATUS_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_full_longrun_status
TRAIN_50K_ROOT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_50k_training/tflop_fintabnet_train_20k/from_ftn10k_checkpoint_50000steps_trainonly
TRAIN_OUT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_full_100k_training

if [ -f "$STATUS_DIR/SKIPPED" ]; then
  echo "Full-train gate skipped; not starting 100K run."
  exit 0
fi
test -f "$STATUS_DIR/READY"
test -f "$DATA_DIR/data_config.yaml"

MODEL_50K=$($PY - <<'PY'
from pathlib import Path
import re

root = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_50k_training/tflop_fintabnet_train_20k/from_ftn10k_checkpoint_50000steps_trainonly")
candidates = []
for path in root.glob("epoch_*_step_*"):
    if not (path / "pytorch_model.bin").exists() or not (path / "config.json").exists():
        continue
    match = re.search(r"_step_(\d+)$", path.name)
    step = int(match.group(1)) if match else -1
    candidates.append((step, path))
if not candidates:
    raise SystemExit(f"No 50K checkpoint found under {root}")
print(max(candidates, key=lambda item: item[0])[1])
PY
)

$PY_TFLOP scripts/train_tflop_single_gpu_train_only.py \
  --exp_config /cluster/home/trinhwin/vt2/docling/repo/TFLOP/config/exp_configs/general_exp.yaml \
  --data_config "$DATA_DIR/data_config.yaml" \
  exp_name=tflop_fintabnet_full_train \
  exp_version=from_ftn50k_checkpoint_100000steps_trainonly \
  result_path="$TRAIN_OUT" \
  pretrained_tokenizer_name_or_path="$MODEL_50K" \
  pretrained_model_name_or_path="$MODEL_50K" \
  max_length=1376 \
  max_position_embeddings=1376 \
  bbox_token_cnt=640 \
  train_batch_size=2 \
  val_batch_size=2 \
  num_workers=8 \
  lr=0.00008 \
  max_steps=100000 \
  max_epochs=-1 \
  save_every_n_train_steps=20000 \
  val_check_interval=1000 \
  check_val_every_n_epoch=1 \
  use_OTSL=True \
  use_imgRoiAlign=True \
  use_RowWise_contLearning=True \
  use_ColWise_contLearning=True \
  use_bbox_HiMulConET=True \
  use_ptr_decoder=True \
  empty_cell_ptr_loss_coeff=0.5 \
  non_empty_cell_ptr_loss_coeff=0.5 \
  strategy=auto \
  precision=16-mixed \
  num_sanity_val_steps=0
