#!/bin/bash
#SBATCH --job-name=tflop_ptn_test
#SBATCH --time=24:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu_top
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs results/tflop_pubtabnet_test_9064_official_full_rerun

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
TFLOP_DATA=/cluster/home/trinhwin/vt2/docling/data/TFLOP-dataset
SAVE_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_test_9064_official_full_rerun

$PY $TFLOP_REPO/test.py \
  --tokenizer_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --model_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --exp_config_path /cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_official_repro/config.yaml \
  --model_config_path /cluster/home/trinhwin/vt2/docling/models/tflop/config.json \
  --aux_json_path $TFLOP_DATA/meta_data/final_eval_v2.json \
  --aux_img_path $TFLOP_DATA/images/test \
  --aux_rec_pkl_path $TFLOP_DATA/pse_results/test/end2end_results.pkl \
  --batch_size 1 \
  --save_dir $SAVE_DIR

$PY $TFLOP_REPO/evaluate_ted.py \
  --model_inference_pathdir $SAVE_DIR \
  --output_savepath $SAVE_DIR
