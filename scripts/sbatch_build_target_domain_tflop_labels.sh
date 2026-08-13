#!/usr/bin/env bash
#SBATCH --job-name=target_domain_label_gate
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail
ROOT=/cluster/home/trinhwin/vt2/docling
OUT="$ROOT/results/tflop_target_domain_train_pseudo_labels"
cd "$ROOT"
rm -rf "$OUT"
.venv/bin/python scripts/build_target_domain_tflop_training_dataset.py \
    --inventory-dir results/target_domain_train_inventory_split \
    --ocr-pkl results/target_domain_train_psenet_master/aux_rec.pkl \
    --output-dir "$OUT"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("results/tflop_target_domain_train_pseudo_labels/summary.json").read_text())
train = summary["splits"]["train"]["accepted"]
validation = summary["splits"]["validation"]["accepted"]
if train < 2000:
    raise SystemExit(f"Quality gate failed: only {train} accepted TargetDomain training tables")
if validation < 200:
    raise SystemExit(f"Quality gate failed: only {validation} accepted TargetDomain validation tables")
if summary["mean_character_coverage"] < 0.78:
    raise SystemExit("Quality gate failed: mean character coverage below 0.78")
if summary["mean_weighted_similarity"] < 0.82:
    raise SystemExit("Quality gate failed: mean weighted similarity below 0.82")
Path("results/tflop_target_domain_train_pseudo_labels/QUALITY_GATE_PASSED").write_text(
    f"train={train}\nvalidation={validation}\n"
)
print("TargetDomain pseudo-label quality gate passed")
PY
