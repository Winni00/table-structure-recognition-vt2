#!/bin/bash
#SBATCH --job-name=ftn50k_teds_merge
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
RUN_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn50k_eval
BASELINE_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages
FTN1K_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn1k_eval
FTN10K_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn10k_eval

if [ -f "$RUN_DIR/SKIPPED" ]; then
  cat > "$RUN_DIR/comparison_public_1k_10k_50k_summary.json" <<'JSON'
{"status": "skipped", "reason": "50K training gate was not passed"}
JSON
  echo "50K eval skipped; wrote skipped summary."
  exit 0
fi

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
ftn1k = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn1k_eval")
ftn10k = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn10k_eval")
ftn50k = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages_ftn50k_eval")
out = ftn50k / "comparison_public_1k_10k_50k_summary.json"
readme = ftn50k / "README.md"

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
    return json.loads((path / "ted_score_output_ftn_canonicalized.summary.json").read_text(encoding="utf-8"))

summary = {
    "baseline_public_checkpoint_full_ftn_annotation": {
        "official_tflop_eval": official_summary(baseline),
        "ftn_section_canonicalized": canonical_summary(baseline),
    },
    "finetuned_1k_checkpoint_full_ftn_annotation": {
        "official_tflop_eval": official_summary(ftn1k),
        "ftn_section_canonicalized": canonical_summary(ftn1k),
    },
    "finetuned_10k_checkpoint_full_ftn_annotation": {
        "official_tflop_eval": official_summary(ftn10k),
        "ftn_section_canonicalized": canonical_summary(ftn10k),
    },
    "finetuned_50k_checkpoint_full_ftn_annotation": {
        "official_tflop_eval": official_summary(ftn50k),
        "ftn_section_canonicalized": canonical_summary(ftn50k),
    },
}
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def fmt(row: dict) -> str:
    samples = row.get("samples", row.get("num_samples"))
    return f"{samples} | {row['teds_s']:.4f} | {row['teds']:.4f} | {row['teds_s_1']} | {row['teds_1']}"

lines = [
    "# FTN 50K Fine-tune Full-Val Evaluation",
    "",
    "| Run | Eval | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
]
for name, block in summary.items():
    pretty = name.replace("_", " ")
    lines.append(f"| {pretty} | official TFLOP | {fmt(block['official_tflop_eval'])} |")
    lines.append(f"| {pretty} | FTN-section canonicalized | {fmt(block['ftn_section_canonicalized'])} |")
readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
