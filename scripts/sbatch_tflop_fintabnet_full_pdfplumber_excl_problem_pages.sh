#!/bin/bash
#SBATCH --job-name=tflop_fintabnet_pdf_full
#SBATCH --time=1-00:00:00
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
mkdir -p logs results/tflop_fintabnet_full_pdfplumber_excl_problem_pages

PY_BUILD=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
PY_TFLOP=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_pdfplumber_excl_problem_pages

$PY_BUILD /cluster/home/trinhwin/vt2/docling/build_tflop_fintabnet_smoke_aux.py \
  --table-jsonl /cluster/home/trinhwin/vt2/docling/data/fintabnet_kaggle/FinTabNet_1.0.0_cell_val_excluding_28.jsonl \
  --pdf-dir /cluster/home/trinhwin/vt2/docling/data/fintabnet_kaggle/pdfs \
  --output-dir $OUT_DIR \
  --limit 10622 \
  --render-scale 2.0 \
  --use-pdfplumber-layout

$PY_TFLOP $TFLOP_REPO/test.py \
  --tokenizer_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --model_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --exp_config_path /cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_official_repro/config.yaml \
  --model_config_path /cluster/home/trinhwin/vt2/docling/models/tflop/config.json \
  --aux_json_path $OUT_DIR/aux.json \
  --aux_img_path $OUT_DIR/images \
  --aux_rec_pkl_path $OUT_DIR/aux_rec.pkl \
  --batch_size 1 \
  --save_dir $OUT_DIR \
  --use_validation

$PY_TFLOP $TFLOP_REPO/evaluate_ted.py \
  --model_inference_pathdir $OUT_DIR \
  --output_savepath $OUT_DIR
