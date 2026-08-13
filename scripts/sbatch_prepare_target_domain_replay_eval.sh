#!/usr/bin/env bash
#SBATCH --job-name=target_domain_eval_prepare
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail
ROOT=/cluster/home/trinhwin/vt2/docling
cd "$ROOT"
LABELS="$ROOT/results/tflop_target_domain_train_pseudo_labels"
OUT="$ROOT/results/tflop_target_domain_val_eval_input"
test -f "$LABELS/QUALITY_GATE_PASSED"

.venv/bin/python scripts/build_tflop_eval_aux_from_dataset_jsonl.py \
  --dataset-jsonl "$LABELS/meta_data/dataset_validation.jsonl" \
  --source-image-dir "$LABELS/images/validation" \
  --output-dir "$OUT" \
  --force

test -s "$OUT/aux.json"
test -s "$OUT/aux_rec.pkl"
touch "$OUT/READY"
