#!/bin/bash
#SBATCH --job-name=ftn10kptn_teds_merge
#SBATCH --time=30:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
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
RUN_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn10k_ptnstyle_eval

$PY scripts/merge_ted_shards.py \
  --run-dir "$RUN_DIR" \
  --num-shards 16 \
  --output-name ted_score_output.json

$PY scripts/merge_fintabnet_canonicalized_ted_shards.py \
  --run-dir "$RUN_DIR" \
  --num-shards 16 \
  --output-name ted_score_output_ftn_canonicalized.json

$PY - <<'PY'
import json
from pathlib import Path

run = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn10k_ptnstyle_eval")
plain = json.loads((run / "ted_score_output.summary.json").read_text())
canon = json.loads((run / "ted_score_output_ftn_canonicalized.summary.json").read_text())
summary = {
    "run": "FTN 10K PTN-style-section training labels",
    "official_tflop_eval": plain,
    "ftn_section_canonicalized": canon,
}
(run / "README.md").write_text(
    "# FTN 10K PTN-style-section Eval\n\n"
    "| Eval | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |\n"
    "|---|---:|---:|---:|---:|---:|\n"
    f"| Official TFLOP | {plain['num_samples']} | {plain['teds_s']:.4f} | {plain['teds']:.4f} | {plain['teds_s_1']} | {plain['teds_1']} |\n"
    f"| FTN-section canonicalized | {canon['num_samples']} | {canon['teds_s']:.4f} | {canon['teds']:.4f} | {canon['teds_s_1']} | {canon['teds_1']} |\n",
    encoding="utf-8",
)
(run / "two_version_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
