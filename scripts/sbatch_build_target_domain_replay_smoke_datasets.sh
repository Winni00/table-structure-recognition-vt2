#!/usr/bin/env bash
#SBATCH --job-name=target_domain_replay_build
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail
ROOT=/cluster/home/trinhwin/vt2/docling
cd "$ROOT"
test -f results/tflop_target_domain_train_pseudo_labels/QUALITY_GATE_PASSED

.venv/bin/python scripts/build_tflop_mixed_target_domain_ptn_dataset.py \
    --out-dir results/tflop_mixed_target_domain2k_ptn667 \
    --target_domain-train-samples 2000 \
    --ptn-train-samples 667 \
    --target_domain-val-samples 200 \
    --ptn-val-samples 67 \
    --force

.venv/bin/python scripts/build_tflop_mixed_target_domain_ptn_dataset.py \
    --out-dir results/tflop_mixed_target_domain2k_ptn2k \
    --target_domain-train-samples 2000 \
    --ptn-train-samples 2000 \
    --target_domain-val-samples 200 \
    --ptn-val-samples 200 \
    --force
