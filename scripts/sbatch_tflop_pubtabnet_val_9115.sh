#!/bin/bash
#SBATCH --job-name=tflop_pubtabnet_val
#SBATCH --time=24:00:00
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
mkdir -p logs results/tflop_pubtabnet_val_9115

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
TFLOP_DATA=/cluster/home/trinhwin/vt2/docling/data/TFLOP-dataset
SAVE_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_val_9115

$PY $TFLOP_REPO/test.py \
  --tokenizer_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --model_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --exp_config_path /cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_official_repro/config.yaml \
  --model_config_path /cluster/home/trinhwin/vt2/docling/models/tflop/config.json \
  --aux_json_path $TFLOP_DATA/meta_data/pubtabnet_val_aux.json \
  --aux_img_path $TFLOP_DATA/images/validation \
  --aux_rec_pkl_path $TFLOP_DATA/pse_results/val/detection_results_0.pkl \
  --batch_size 1 \
  --save_dir $SAVE_DIR \
  --use_validation

$PY $TFLOP_REPO/evaluate_ted.py \
  --model_inference_pathdir $SAVE_DIR \
  --output_savepath $SAVE_DIR
