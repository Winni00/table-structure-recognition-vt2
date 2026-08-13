#!/usr/bin/env bash
#SBATCH --job-name=target_domainupd_label_gate
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail
ROOT=/cluster/home/trinhwin/vt2/docling
OUT="$ROOT/results/tflop_target_domain_train_updated_final_pseudo_labels"
cd "$ROOT"
test -f results/target_domain_train_updated_final_psenet_master/OCR_COMPLETE
rm -rf "$OUT"
.venv/bin/python scripts/build_target_domain_tflop_training_dataset.py \
    --inventory-dir results/target_domain_train_updated_final_inventory_split \
    --ocr-pkl results/target_domain_train_updated_final_psenet_master/aux_rec.pkl \
    --output-dir "$OUT"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("results/tflop_target_domain_train_updated_final_pseudo_labels")
summary = json.loads((root / "summary.json").read_text())
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
(root / "QUALITY_GATE_PASSED").write_text(f"train={train}\nvalidation={validation}\n")
print("TargetDomain updated-final pseudo-label quality gate passed")
print(json.dumps(summary, indent=2)[:4000])
PY
