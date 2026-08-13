#!/bin/bash
#SBATCH --job-name=ftn_mix10k_teds_merge
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
RUN_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_mixed10k_lr1e5_eval

$PY scripts/merge_ted_shards.py --run-dir "$RUN_DIR" --num-shards 16 --output-name ted_score_output.json
$PY scripts/merge_fintabnet_canonicalized_ted_shards.py --run-dir "$RUN_DIR" --num-shards 16 --output-name ted_score_output_ftn_canonicalized.json

$PY - <<'PY'
import json
from pathlib import Path

run = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_mixed10k_lr1e5_eval")

def official(path: Path) -> dict:
    rows = json.loads((path / "ted_score_output.json").read_text(encoding="utf-8"))
    return {
        "samples": len(rows),
        "teds_s": sum(r[-2] for r in rows) / len(rows),
        "teds": sum(r[-1] for r in rows) / len(rows),
        "teds_s_1": sum(r[-2] == 1.0 for r in rows),
        "teds_1": sum(r[-1] == 1.0 for r in rows),
    }

def canon(path: Path) -> dict:
    return json.loads((path / "ted_score_output_ftn_canonicalized.summary.json").read_text(encoding="utf-8"))

summary = {
    "model_variant": "mixed FTN16K+PTN4K, 10K steps, LR=1e-5",
    "official_tflop_eval": official(run),
    "ftn_section_canonicalized": canon(run),
}
(run / "two_version_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(run / "README.md").write_text(
    "# FTN Full Eval: Mixed FTN+PTN 10K LR=1e-5\n\n"
    "| Eval | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |\n"
    "|---|---:|---:|---:|---:|---:|\n"
    f"| Official TFLOP | {summary['official_tflop_eval']['samples']} | {summary['official_tflop_eval']['teds_s']:.4f} | {summary['official_tflop_eval']['teds']:.4f} | {summary['official_tflop_eval']['teds_s_1']} | {summary['official_tflop_eval']['teds_1']} |\n"
    f"| FTN-section canonicalized | {summary['ftn_section_canonicalized']['num_samples']} | {summary['ftn_section_canonicalized']['teds_s']:.4f} | {summary['ftn_section_canonicalized']['teds']:.4f} | {summary['ftn_section_canonicalized']['teds_s_1']} | {summary['ftn_section_canonicalized']['teds_1']} |\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
PY
