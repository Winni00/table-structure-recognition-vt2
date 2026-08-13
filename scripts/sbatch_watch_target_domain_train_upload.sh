#!/usr/bin/env bash
#SBATCH --job-name=target_domain_upload_watch
#SBATCH --partition=cpu_low
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-00:00:00
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/target_domain_upload_watch_%j.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/target_domain_upload_watch_%j.err

set -euo pipefail

ROOT=/cluster/home/trinhwin/vt2/docling
mkdir -p "$ROOT/logs"
exec bash "$ROOT/scripts/watch_target_domain_train_upload.sh"
