#!/bin/bash
#SBATCH --job-name=tflop_mix10k_lr1e5
#SBATCH --time=12:00:00
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

PY_TFLOP=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
DATA_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_mixed_ftn16k_ptn4k_for_10k
TRAIN_OUT=/cluster/home/trinhwin/vt2/docling/results/tflop_mixed_ftn16k_ptn4k_train_10k_lr1e5

test -f "$DATA_DIR/data_config.yaml"
test -f "$DATA_DIR/meta_data/dataset_train.jsonl"
test -f "$DATA_DIR/meta_data/dataset_validation.jsonl"

$PY_TFLOP scripts/train_tflop_single_gpu_train_only.py \
  --exp_config /cluster/home/trinhwin/vt2/docling/repo/TFLOP/config/exp_configs/general_exp.yaml \
  --data_config "$DATA_DIR/data_config.yaml" \
  exp_name=tflop_mixed_ftn16k_ptn4k \
  exp_version=from_public_checkpoint_10000steps_lr1e5 \
  result_path="$TRAIN_OUT" \
  pretrained_tokenizer_name_or_path=/cluster/home/trinhwin/vt2/docling/models/tflop \
  pretrained_model_name_or_path=/cluster/home/trinhwin/vt2/docling/models/tflop \
  max_length=1376 \
  max_position_embeddings=1376 \
  bbox_token_cnt=640 \
  train_batch_size=2 \
  val_batch_size=2 \
  num_workers=8 \
  lr=0.00001 \
  max_steps=10000 \
  max_epochs=-1 \
  save_every_n_train_steps=5000 \
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

echo "Finished mixed FTN+PTN 10K LR=1e-5 fine-tune in $TRAIN_OUT"
