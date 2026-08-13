#!/bin/bash
#SBATCH --job-name=tflop_ftn_train1k
#SBATCH --time=08:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY_BUILD=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
PY_TFLOP=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
DATA_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_smoke_1k
TRAIN_OUT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_smoke_1k_training

$PY_BUILD scripts/build_ftn_trainval_tflop_dataset.py \
  --out-dir "$DATA_DIR" \
  --train-samples 1000 \
  --val-samples 200 \
  --max-cells 140 \
  --render-scale 2.0 \
  --bbox-token-cnt 640 \
  --max-length 1376 \
  --force

$PY_TFLOP scripts/train_tflop_single_gpu.py \
  --exp_config /cluster/home/trinhwin/vt2/docling/repo/TFLOP/config/exp_configs/general_exp.yaml \
  --data_config "$DATA_DIR/data_config.yaml" \
  exp_name=tflop_fintabnet_train_smoke_1k \
  exp_version=from_public_checkpoint_1000steps \
  result_path="$TRAIN_OUT" \
  pretrained_tokenizer_name_or_path=/cluster/home/trinhwin/vt2/docling/models/tflop \
  pretrained_model_name_or_path=/cluster/home/trinhwin/vt2/docling/models/tflop \
  max_length=1376 \
  max_position_embeddings=1376 \
  bbox_token_cnt=640 \
  train_batch_size=2 \
  val_batch_size=2 \
  num_workers=8 \
  lr=0.00008 \
  max_steps=1000 \
  max_epochs=-1 \
  val_check_interval=100 \
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
  num_sanity_val_steps=1
