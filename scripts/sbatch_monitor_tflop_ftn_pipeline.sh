#!/bin/bash
#SBATCH --job-name=ftn_pipeline_monitor
#SBATCH --time=20-00:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --account=cai_exp
#SBATCH --partition=cpu_low
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

exec scripts/monitor_tflop_ftn_pipeline.sh \
  "26639,26640,26641,26642,26643,26824,26825,27046,27047" \
  300
