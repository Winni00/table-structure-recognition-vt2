#!/bin/bash
#SBATCH --job-name=tflop_ptn_val_ann8958
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
mkdir -p logs results/tflop_pubtabnet_val_annotations_filtered_8958

PY_BUILD=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
PY_TFLOP=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_val_annotations_filtered_8958

$PY_BUILD /cluster/home/trinhwin/vt2/docling/build_tflop_pubtabnet_annotation_aux.py \
  --jsonl /cluster/home/trinhwin/vt2/docling/data/TFLOP-dataset/meta_data/PubTabNet_2.0.0.jsonl \
  --images-dir /cluster/home/trinhwin/vt2/docling/data/TFLOP-dataset/images/validation \
  --erroneous-json /cluster/home/trinhwin/vt2/docling/data/TFLOP-dataset/meta_data/erroneous_pubtabnet_data.json \
  --output-dir $OUT_DIR \
  --split val

$PY_TFLOP $TFLOP_REPO/test.py \
  --tokenizer_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --model_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --exp_config_path /cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_official_repro/config.yaml \
  --model_config_path /cluster/home/trinhwin/vt2/docling/models/tflop/config.json \
  --aux_json_path $OUT_DIR/aux.json \
  --aux_img_path /cluster/home/trinhwin/vt2/docling/data/TFLOP-dataset/images/validation \
  --aux_rec_pkl_path $OUT_DIR/aux_rec.pkl \
  --batch_size 1 \
  --save_dir $OUT_DIR \
  --use_validation

$PY_TFLOP $TFLOP_REPO/evaluate_ted.py \
  --model_inference_pathdir $OUT_DIR \
  --output_savepath $OUT_DIR
