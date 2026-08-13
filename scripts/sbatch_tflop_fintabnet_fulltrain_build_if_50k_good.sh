#!/bin/bash
#SBATCH --job-name=ftn_full_build_gate
#SBATCH --time=12:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
REPORT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn50k_eval/comparison_public_1k_10k_50k_summary.json
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_full_for_longrun
STATUS_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_full_longrun_status

mkdir -p "$STATUS_DIR"
rm -f "$STATUS_DIR/SKIPPED" "$STATUS_DIR/READY"
test -f "$REPORT"

echo "Full dataset build is forced: continuing regardless of the 50K score."

$PY scripts/build_ftn_trainval_tflop_dataset.py \
  --out-dir "$OUT_DIR" \
  --train-samples 91596 \
  --val-samples 1000 \
  --max-cells 140 \
  --render-scale 2.0 \
  --bbox-token-cnt 640 \
  --max-length 1376 \
  --force

touch "$STATUS_DIR/READY"
echo "Built full FTN train dataset for long run at $OUT_DIR"
