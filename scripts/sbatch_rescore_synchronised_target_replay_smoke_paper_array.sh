#!/usr/bin/env bash
#SBATCH --job-name=zhawupd_rescore
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

ROOT=/cluster/home/trinhwin/vt2/docling
PY="$ROOT/.venv-tflop39/bin/python"
EVAL_ROOT="$ROOT/results/tflop_synchronised_target_replay_smoke_1k_eval"
cd "$ROOT"

case "${SLURM_ARRAY_TASK_ID}" in
  0) VARIANT=public ;;
  1) VARIANT=target_domainupd_only ;;
  2) VARIANT=target_domainupd75_ptn25 ;;
  3) VARIANT=target_domainupd50_ptn50 ;;
  *) exit 2 ;;
esac

RUN_DIR="$EVAL_ROOT/$VARIANT/paper_test"
test -s "$RUN_DIR/full_model_inference.json"

"$PY" scripts/rescore_paper_collection_gt_canonicalized.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/gt_canonicalized_rescore" \
  --num-processes "${SLURM_CPUS_PER_TASK:-8}"

touch "$RUN_DIR/gt_canonicalized_rescore/COMPLETE"
