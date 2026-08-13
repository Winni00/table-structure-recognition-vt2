#!/bin/bash
#SBATCH --job-name=tflop_ptn_val_pse_inf
#SBATCH --time=1-00:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40spcie:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu_inference
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
INPUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_val_pse_matched_filtered_8958
SAVE_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_val_pse_matched_filtered_8958_inference

mkdir -p "$SAVE_DIR"
ln -sfn "$INPUT_DIR/aux.json" "$SAVE_DIR/aux.json"
ln -sfn "$INPUT_DIR/aux_rec.pkl" "$SAVE_DIR/aux_rec.pkl"
cp "$INPUT_DIR/summary.json" "$SAVE_DIR/input_summary.json"

$PY $TFLOP_REPO/test.py \
  --tokenizer_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --model_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --exp_config_path /cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_official_repro/config.yaml \
  --model_config_path /cluster/home/trinhwin/vt2/docling/models/tflop/config.json \
  --aux_json_path "$INPUT_DIR/aux.json" \
  --aux_img_path /cluster/home/trinhwin/vt2/docling/data/TFLOP-dataset/images/validation \
  --aux_rec_pkl_path "$INPUT_DIR/aux_rec.pkl" \
  --batch_size 1 \
  --save_dir "$SAVE_DIR" \
  --use_validation

$PY /cluster/home/trinhwin/vt2/docling/repo/TFLOP_clean/evaluate_ted.py \
  --model_inference_pathdir "$SAVE_DIR" \
  --output_savepath "$SAVE_DIR"
