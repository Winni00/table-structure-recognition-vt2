#!/usr/bin/env bash
#SBATCH --job-name=target_domainupd_eval_prep
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail
ROOT=/cluster/home/trinhwin/vt2/docling
LABELS="$ROOT/results/tflop_target_domain_train_updated_final_pseudo_labels"
OUT="$ROOT/results/tflop_synchronised_target_val_eval_input"
cd "$ROOT"
test -f "$LABELS/QUALITY_GATE_PASSED"
rm -rf "$OUT"
.venv/bin/python scripts/build_tflop_eval_aux_from_dataset_jsonl.py \
  --dataset-jsonl "$LABELS/meta_data/dataset_validation.jsonl" \
  --source-image-dir "$LABELS/images/validation" \
  --output-dir "$OUT" \
  --force
touch "$OUT/READY"
