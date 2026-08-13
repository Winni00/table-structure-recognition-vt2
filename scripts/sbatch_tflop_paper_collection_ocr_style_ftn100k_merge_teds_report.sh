#!/bin/bash
#SBATCH --job-name=paper_ftn100k_teds
#SBATCH --time=2:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=cai_exp
#SBATCH --partition=cpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
PY_SYS=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
SOURCE_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_ocr_style_707
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_ocr_style_707_ftn100k

ln -sfn "$SOURCE_DIR/images" "$OUT_DIR/images"
ln -sfn "$SOURCE_DIR/aux.json" "$OUT_DIR/aux.json"
ln -sfn "$SOURCE_DIR/aux_rec.pkl" "$OUT_DIR/aux_rec.pkl"
ln -sfn "$SOURCE_DIR/manifest.json" "$OUT_DIR/manifest.json"
ln -sfn "$SOURCE_DIR/skipped_samples.json" "$OUT_DIR/skipped_samples.json"
ln -sfn "$SOURCE_DIR/post_ocr_validation.json" "$OUT_DIR/post_ocr_validation.json"

$PY_SYS scripts/merge_tflop_inference_shards.py \
  --shard-dir "$OUT_DIR/inference_shards" \
  --output-json "$OUT_DIR/full_model_inference.json" \
  --summary-json "$OUT_DIR/inference_merge_summary.json"

for SHARD in $(seq 0 7); do
  $PY scripts/evaluate_ted_shard.py \
    --run-dir "$OUT_DIR" \
    --shard-index "$SHARD" \
    --num-shards 8 \
    --num-processes 8 \
    --batch-size 50
done

$PY_SYS scripts/merge_ted_shards.py \
  --run-dir "$OUT_DIR" \
  --num-shards 8 \
  --output-name ted_score_output.json

$PY_SYS scripts/report_paper_collection_ocr_style_results.py \
  --run-dir "$OUT_DIR" \
  --output-dir "$OUT_DIR/report_no_gt_canonicalization" \
  --examples-per-group 5
ln -sfn "$OUT_DIR/report_no_gt_canonicalization" "$OUT_DIR/report_readable"

$PY scripts/rescore_paper_collection_gt_canonicalized.py \
  --run-dir "$OUT_DIR" \
  --output-dir "$OUT_DIR/gt_canonicalized_rescore" \
  --num-processes 8 \
  --batch-size 50

$PY_SYS - <<'PY'
import json
from pathlib import Path

out_dir = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_paper_collection_ocr_style_707_ftn100k")
plain = json.loads((out_dir / "ted_score_output.summary.json").read_text())
canon = json.loads((out_dir / "gt_canonicalized_rescore/ted_score_output.summary.json").read_text())
summary = {
    "run_dir": str(out_dir),
    "model_variant": "FTN-100K checkpoint on paper collection OCR-style input",
    "same_for_both_versions": [
        "same TFLOP predictions",
        "same PSENet+MASTER OCR input",
        "same official TFLOP TEDS implementation",
    ],
    "versions": {
        "without_gt_canonicalization": {
            "description": "Official/no extra GT canonicalization; publisher GT HTML is scored as originally provided.",
            "report_dir": str(out_dir / "report_no_gt_canonicalization"),
            **plain,
        },
        "with_gt_canonicalization": {
            "description": "Same predictions, but GT HTML is converted toward a PubTabNet/TFLOP-friendly style before rescoring.",
            "report_dir": str(out_dir / "gt_canonicalized_rescore"),
            **canon,
        },
    },
}
(out_dir / "two_version_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
md = [
    "# Paper Collection FTN-100K: Two Evaluation Versions",
    "",
    "Both versions use the same predictions and the same PSENet+MASTER OCR input. Only the GT HTML treatment differs.",
    "",
    "| Version | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 | Output |",
    "|---|---:|---:|---:|---:|---:|---|",
    f"| Without GT canonicalization | {plain['num_samples']} | {plain['teds_s']:.4f} | {plain['teds']:.4f} | {plain['teds_s_1']} | {plain['teds_1']} | `report_no_gt_canonicalization` |",
    f"| With GT canonicalization | {canon['num_samples']} | {canon['teds_s']:.4f} | {canon['teds']:.4f} | {canon['teds_s_1']} | {canon['teds_1']} | `gt_canonicalized_rescore` |",
    "",
    "Use the first row for strict/original-GT reporting, and the second row to estimate how much score is lost only because of publisher-style GT HTML.",
]
(out_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
PY

echo "Paper collection FTN-100K inference merged, evaluated, and reported."
