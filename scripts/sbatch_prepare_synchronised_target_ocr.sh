#!/usr/bin/env bash
#SBATCH --job-name=target_domainupd_prepare
#SBATCH --partition=cpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail
ROOT=/cluster/home/trinhwin/vt2/docling
ZIP="$ROOT/data/train_updated_final.zip"
EXTRACT="$ROOT/data/target_domain_train_updated_final"
RAW="$EXTRACT/train"
INV="$ROOT/results/target_domain_train_updated_final_inventory_split"
OCR="$ROOT/results/target_domain_train_updated_final_psenet_master"
cd "$ROOT"
mkdir -p logs

test -s "$ZIP"
zip -T "$ZIP"

rm -rf "$EXTRACT" "$INV" "$OCR"
mkdir -p "$EXTRACT"
unzip -q "$ZIP" -d "$EXTRACT" -x "__MACOSX/*" "*/._*"
test -d "$RAW"

.venv/bin/python scripts/inventory_split_target_domain_train.py \
    --data-dir "$RAW" \
    --output-dir "$INV"

.venv/bin/python scripts/prepare_tflop_paper_collection_ocr_style_inputs.py \
    --raw-root "$RAW" \
    --output-dir "$OCR"

.venv/bin/python scripts/split_text_file_shards.py \
    "$OCR/subset.txt" "$OCR/ocr_shards" --num-shards 16 --prefix subset

.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("/cluster/home/trinhwin/vt2/docling")
inv = json.loads((root / "results/target_domain_train_updated_final_inventory_split/summary.json").read_text())
ocr_lines = (root / "results/target_domain_train_updated_final_psenet_master/subset.txt").read_text().splitlines()
print("Inventory summary:")
print(json.dumps(inv, indent=2)[:4000])
print(f"OCR candidates: {len(ocr_lines)}")
PY

touch "$OCR/PREP_COMPLETE"
