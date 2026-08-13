#!/usr/bin/env python3
"""Aggregate synchronized ZHAW replay smoke evaluations."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
EVAL = ROOT / "results/tflop_synchronised_target_replay_smoke_1k_eval"
VARIANTS = ("public", "target_domainupd_only", "target_domainupd75_ptn25", "target_domainupd50_ptn50")
DATASETS = ("target_domain_val", "ptn_val_pse")
VARIANT_LABELS = {
    "public": "Public TFLOP",
    "target_domainupd_only": "ZHAW-only 1K",
    "target_domainupd75_ptn25": "ZHAW 75% + PTN 25%",
    "target_domainupd50_ptn50": "ZHAW 50% + PTN 50%",
}
DATASET_LABELS = {
    "target_domain_val": "Synchronized ZHAW validation",
    "ptn_val_pse": "PubTabNet validation (PSE-matched)",
    "paper_test_raw": "ZHAW paper collection (original reference)",
    "paper_test_gt_canon": "ZHAW paper collection (normalized reference)",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_load(path: Path) -> dict | None:
    return load(path) if path.is_file() else None


def main() -> None:
    rows = []
    for variant in VARIANTS:
        for dataset in DATASETS:
            summary = load(EVAL / variant / dataset / "ted_score_output.summary.json")
            rows.append({"variant": variant, "dataset": dataset, **summary})
        paper_raw = maybe_load(EVAL / variant / "paper_test/ted_score_output.summary.json")
        if paper_raw is not None:
            rows.append({"variant": variant, "dataset": "paper_test_raw", **paper_raw})
        canon = maybe_load(
            EVAL
            / variant
            / "paper_test/gt_canonicalized_rescore/ted_score_output.summary.json"
        )
        if canon is not None:
            rows.append({"variant": variant, "dataset": "paper_test_gt_canon", **canon})

    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Synchronized ZHAW Replay Fine-tuning: 1K Smoke Comparison",
        "",
        "All trained variants start from the public TFLOP checkpoint and use LR=1e-5.",
        "",
        "| Variant | Evaluation | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        samples = row.get("samples", row.get("num_samples", 0))
        lines.append(
            f"| {VARIANT_LABELS[row['variant']]} | "
            f"{DATASET_LABELS[row['dataset']]} | {samples} | "
            f"{row['teds_s']:.4f} | {row['teds']:.4f} | "
            f"{row['teds_s_1']} | {row['teds_1']} |"
        )
    (EVAL / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
