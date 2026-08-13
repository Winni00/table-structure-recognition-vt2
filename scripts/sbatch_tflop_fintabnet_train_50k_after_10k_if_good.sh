#!/bin/bash
#SBATCH --job-name=tflop_ftn50k_gate
#SBATCH --time=2-00:00:00
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
DATA_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_20k_for_10k
TRAIN_10K_ROOT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_10k_training/tflop_fintabnet_train_20k/from_public_checkpoint_10000steps_trainonly
TRAIN_OUT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_50k_training
REPORT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn10k_eval/comparison_public_1k_10k_summary.json

test -f "$DATA_DIR/data_config.yaml"
test -f "$REPORT"

SHOULD_RUN_AND_MODEL=$($PY - <<'PY'
import json
import re
from pathlib import Path

report = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn10k_eval/comparison_public_1k_10k_summary.json")
root = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_10k_training/tflop_fintabnet_train_20k/from_public_checkpoint_10000steps_trainonly")

summary = json.loads(report.read_text(encoding="utf-8"))
block = summary["finetuned_10k_checkpoint_full_ftn_annotation"]
official = block["official_tflop_eval"]
canon = block["ftn_section_canonicalized"]

# Conservative gate: continue only if 10K is at least clearly useful.
if official["teds"] < 0.88 or canon["teds"] < 0.93:
    print("SKIP")
    raise SystemExit(0)

candidates = []
for path in root.glob("epoch_*_step_*"):
    if not (path / "pytorch_model.bin").exists() or not (path / "config.json").exists():
        continue
    match = re.search(r"_step_(\d+)$", path.name)
    step = int(match.group(1)) if match else -1
    candidates.append((step, path))
if not candidates:
    raise SystemExit(f"No 10K checkpoint found under {root}")
print(max(candidates, key=lambda item: item[0])[1])
PY
)

if [ "$SHOULD_RUN_AND_MODEL" = "SKIP" ]; then
  echo "10K result did not pass continuation gate; skipping 50K training."
  exit 0
fi

MODEL_10K="$SHOULD_RUN_AND_MODEL"
echo "Continuing FTN fine-tuning from 10K checkpoint: $MODEL_10K"

$PY_TFLOP scripts/train_tflop_single_gpu_train_only.py \
  --exp_config /cluster/home/trinhwin/vt2/docling/repo/TFLOP/config/exp_configs/general_exp.yaml \
  --data_config "$DATA_DIR/data_config.yaml" \
  exp_name=tflop_fintabnet_train_20k \
  exp_version=from_ftn10k_checkpoint_50000steps_trainonly \
  result_path="$TRAIN_OUT" \
  pretrained_tokenizer_name_or_path="$MODEL_10K" \
  pretrained_model_name_or_path="$MODEL_10K" \
  max_length=1376 \
  max_position_embeddings=1376 \
  bbox_token_cnt=640 \
  train_batch_size=2 \
  val_batch_size=2 \
  num_workers=8 \
  lr=0.00008 \
  max_steps=50000 \
  max_epochs=-1 \
  save_every_n_train_steps=10000 \
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

echo "Finished gated 50K FTN fine-tune in $TRAIN_OUT"
