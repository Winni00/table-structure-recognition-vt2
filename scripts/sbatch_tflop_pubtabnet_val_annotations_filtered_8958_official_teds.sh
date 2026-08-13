#!/bin/bash
#SBATCH --job-name=teds_ptn_val_tflop_official
#SBATCH --time=6:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY_TFLOP=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
RUN_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_val_annotations_filtered_8958
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_val_annotations_filtered_8958_tflop_official_eval

mkdir -p "$OUT_DIR"

if [[ ! -f "$RUN_DIR/full_model_inference.json" ]]; then
  echo "Missing full_model_inference.json in $RUN_DIR" >&2
  exit 1
fi

PYTHONPATH=/cluster/home/trinhwin/vt2/docling/repo/TFLOP \
  "$PY_TFLOP" /cluster/home/trinhwin/vt2/docling/scripts/evaluate_ted_upstage_official.py \
  --model_inference_pathdir "$RUN_DIR" \
  --output_savepath "$OUT_DIR"

"$PY_TFLOP" - <<'PY'
import json
from pathlib import Path

out = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_val_annotations_filtered_8958_tflop_official_eval")
rows = json.loads((out / "ted_score_output.json").read_text(encoding="utf-8"))
summary = {
    "samples": len(rows),
    "teds_s": sum(r[-2] for r in rows) / len(rows),
    "teds": sum(r[-1] for r in rows) / len(rows),
    "teds_s_1": sum(r[-2] == 1.0 for r in rows),
    "teds_1": sum(r[-1] == 1.0 for r in rows),
    "evaluator": "UpstageAI/TFLOP official evaluate_ted.py",
    "normalization": "wrap html + always strip_html_contents(pred/gt)",
    "full_teds_ignore_nodes": ["b"],
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
