#!/usr/bin/env python3
"""Aggregate TargetDomain replay smoke evaluations and compare against baselines."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
EVAL = ROOT / "results/tflop_target_domain_replay_smoke_1k_eval"
VARIANTS = ("public", "target_domain_only", "target_domain75_ptn25", "target_domain50_ptn50")
DATASETS = ("target_domain_val", "ptn_val_pse")


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

    (EVAL / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# TargetDomain Replay Fine-tuning: 1K Smoke Comparison",
        "",
        "All variants start from the public TFLOP checkpoint and use LR=1e-5.",
        "",
        "| Variant | Evaluation | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        samples = row.get("samples", row.get("num_samples", 0))
        lines.append(
            f"| {row['variant']} | {row['dataset']} | {samples} | "
            f"{row['teds_s']:.4f} | {row['teds']:.4f} | "
            f"{row['teds_s_1']} | {row['teds_1']} |"
        )
    (EVAL / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
