#!/bin/bash
#SBATCH --job-name=ftn1k_teds_merge
#SBATCH --time=30:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
RUN_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn1k_eval
BASELINE_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages

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

baseline = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages")
finetuned = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn1k_eval")
out = finetuned / "comparison_with_public_baseline_summary.json"

def official_summary(path: Path) -> dict:
    rows = json.loads((path / "ted_score_output.json").read_text(encoding="utf-8"))
    return {
        "samples": len(rows),
        "teds_s": sum(row[-2] for row in rows) / len(rows),
        "teds": sum(row[-1] for row in rows) / len(rows),
        "teds_s_1": sum(row[-2] == 1.0 for row in rows),
        "teds_1": sum(row[-1] == 1.0 for row in rows),
    }

def canonical_summary(path: Path) -> dict:
    summary_path = path / "ted_score_output_ftn_canonicalized.summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))

summary = {
    "baseline_public_checkpoint_full_ftn_annotation": {
        "official_tflop_eval": official_summary(baseline),
        "ftn_section_canonicalized": canonical_summary(baseline),
    },
    "finetuned_1k_checkpoint_full_ftn_annotation": {
        "official_tflop_eval": official_summary(finetuned),
        "ftn_section_canonicalized": canonical_summary(finetuned),
    },
}
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "Merged FTN 1K fine-tuned full annotation TEDS shards and wrote comparison summary."
