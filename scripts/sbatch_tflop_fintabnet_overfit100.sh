#!/bin/bash
#SBATCH --job-name=tflop_ftn_overfit100
#SBATCH --time=06:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs results/tflop_fintabnet_overfit100_training

PY_BUILD=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
PY_TFLOP=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python

$PY_BUILD scripts/build_ftn_overfit_tflop_dataset.py \
  --out-dir /cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_overfit100 \
  --num-samples 100 \
  --max-cells 140 \
  --render-scale 2.0

$PY_TFLOP scripts/train_tflop_single_gpu.py \
  --exp_config /cluster/home/trinhwin/vt2/docling/repo/TFLOP/config/exp_configs/general_exp.yaml \
  --data_config /cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_overfit100/data_config.yaml \
  exp_name=tflop_fintabnet_overfit100 \
  exp_version=from_public_checkpoint_300steps \
  result_path=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_overfit100_training \
  pretrained_tokenizer_name_or_path=/cluster/home/trinhwin/vt2/docling/models/tflop \
  pretrained_model_name_or_path=/cluster/home/trinhwin/vt2/docling/models/tflop \
  max_length=1376 \
  max_position_embeddings=1376 \
  bbox_token_cnt=640 \
  train_batch_size=2 \
  val_batch_size=2 \
  num_workers=4 \
  lr=0.00008 \
  max_steps=300 \
  max_epochs=-1 \
  val_check_interval=50 \
  check_val_every_n_epoch=1 \
  use_OTSL=True \
  use_imgRoiAlign=True \
  use_RowWise_contLearning=True \
  use_ColWise_contLearning=True \
  use_bbox_HiMulConET=True \
  strategy=auto \
  precision=16-mixed
