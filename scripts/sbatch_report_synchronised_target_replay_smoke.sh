#!/usr/bin/env bash
#SBATCH --job-name=target_domainupd_report
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=30:00
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /cluster/home/trinhwin/vt2/docling
.venv/bin/python scripts/report_synchronised_target_replay_smoke_results.py
