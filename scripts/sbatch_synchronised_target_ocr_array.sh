#!/usr/bin/env bash
#SBATCH --job-name=target_domainupd_ocr
#SBATCH --partition=gpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-15%4
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=FAIL,TIME_LIMIT

set -euo pipefail
ROOT=/cluster/home/trinhwin/vt2/docling
OUT="$ROOT/results/target_domain_train_updated_final_psenet_master"
PY="$ROOT/.venv-tablemaster38/bin/python"
REPRO="$ROOT/repro/pubtabnet_tflop_reproduction"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
SUBSET=$(printf "%s/ocr_shards/subset_%02d.txt" "$OUT" "$TASK_ID")
OUTPUT=$(printf "%s/ocr_shards/aux_rec_%02d.pkl" "$OUT" "$TASK_ID")
cd "$ROOT"
test -f "$OUT/PREP_COMPLETE"
test -s "$SUBSET"
"$PY" "$REPRO/scripts/run_tablemaster_end2end_subset.py" \
    --subset "$SUBSET" \
    --images-dir "$OUT/images" \
    --output-pkl "$OUTPUT" \
    --on-error empty
