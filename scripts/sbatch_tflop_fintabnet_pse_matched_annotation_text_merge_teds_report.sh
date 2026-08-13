#!/bin/bash
#SBATCH --job-name=ftn_psegt_teds_merge
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
RUN_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_pse_matched_annotation_text_full

$PY scripts/merge_ted_shards.py --run-dir "$RUN_DIR" --num-shards 16 --output-name ted_score_output.json
$PY scripts/merge_fintabnet_canonicalized_ted_shards.py --run-dir "$RUN_DIR" --num-shards 16 --output-name ted_score_output_ftn_canonicalized.json

$PY - <<'PY'
import json
from pathlib import Path

run = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_pse_matched_annotation_text_full")
annotation = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_full_annotation_excl_problem_pages")
ocr = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_ocr_style_full_10622")

def rows_summary(path: Path, name: str) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "samples": len(rows),
        "teds_s": sum(r[-2] for r in rows) / len(rows),
        "teds": sum(r[-1] for r in rows) / len(rows),
        "teds_s_1": sum(r[-2] == 1.0 for r in rows),
        "teds_1": sum(r[-1] == 1.0 for r in rows),
    }

def optional_summary(path: Path, name: str) -> dict | None:
    if not path.exists():
        return None
    return rows_summary(path, name)

summaries = [
    optional_summary(annotation / "ted_score_output.json", "FTN annotation cell boxes/text, official TFLOP eval"),
    optional_summary(annotation / "ted_score_output_ftn_canonicalized.json", "FTN annotation cell boxes/text, FTN-section canonicalized"),
    optional_summary(ocr / "ted_score_output.json", "FTN PSENet+MASTER OCR-style, official TFLOP eval"),
    rows_summary(run / "ted_score_output.json", "FTN PSE boxes + annotation text, official TFLOP eval"),
    rows_summary(run / "ted_score_output_ftn_canonicalized.json", "FTN PSE boxes + annotation text, FTN-section canonicalized"),
]
summaries = [s for s in summaries if s is not None]

report = {
    "purpose": "Compare FTN input styles: annotation cell boxes/text vs PSENet+MASTER OCR vs PSE boxes with annotation-matched text.",
    "run_dir": str(run),
    "matching_summary": json.loads((run / "summary.json").read_text(encoding="utf-8")),
    "scores": summaries,
}
(run / "input_style_comparison_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

lines = [
    "# FTN PSE Boxes + Annotation Text",
    "",
    "This run uses PSENet/PSE text-region boxes, but replaces MASTER-recognized text with FTN annotation text matched by box overlap. It is the FTN analogue of the PubTabNet train/val PSE-matching setup.",
    "",
    "## Scores",
    "",
    "| Run | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |",
    "|---|---:|---:|---:|---:|---:|",
]
for s in summaries:
    lines.append(
        f"| {s['name']} | {s['samples']} | {s['teds_s']:.4f} | {s['teds']:.4f} | {s['teds_s_1']} | {s['teds_1']} |"
    )
ms = report["matching_summary"]
lines += [
    "",
    "## Matching Summary",
    "",
    f"- Samples: {ms['samples']}",
    f"- Dropped empty after matching: {ms['dropped_empty_after_matching']}",
    f"- PSE regions: {ms['total_pse_regions']}",
    f"- Matched regions: {ms['matched_regions']} ({100 * ms['match_rate']:.2f}%)",
    f"- Unmatched PSE regions: {ms['unmatched_pse_regions']}",
    f"- Mean matched regions per sample: {ms['mean_matched_regions_per_sample']:.2f}",
]
(run / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
