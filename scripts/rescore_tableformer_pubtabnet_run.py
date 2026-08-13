#!/usr/bin/env python3
"""Recompute TableFormer PubTabNet per-table TEDS/TEDS-S from saved artifacts."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
TFLOP_REPO = ROOT / "repo" / "TFLOP"
if str(TFLOP_REPO) not in sys.path:
    sys.path.insert(0, str(TFLOP_REPO))

from tflop.evaluator import TEDS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--write-predictions", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args()


def ensure_html_table_document(text: str) -> str:
    cleaned = text.strip()
    lower = cleaned.lower()
    if "<html" in lower:
        return cleaned
    if "<table" in lower:
        return f"<html><body>{cleaned}</body></html>"
    return f"<html><body><table>{cleaned}</table></body></html>"


def normalized_table(prediction: dict[str, Any]) -> dict[str, Any]:
    normalized = prediction.get("normalized_output", {})
    if isinstance(normalized, list):
        return normalized[0] if normalized else {}
    return normalized if isinstance(normalized, dict) else {}


def prediction_html(prediction: dict[str, Any]) -> str:
    normalized = normalized_table(prediction)
    structure = normalized.get("structure", {})
    html = structure.get("html", "")
    return ensure_html_table_document(html)


def ground_truth_html(prediction: dict[str, Any]) -> str:
    ground_truth = prediction.get("ground_truth", {})
    html = ground_truth.get("gt_html", "")
    return ensure_html_table_document(html)


def score_prediction(pred_path_text: str) -> tuple[str, float, float]:
    pred_path = Path(pred_path_text)
    prediction = json.loads(pred_path.read_text(encoding="utf-8"))
    pred_html = prediction_html(prediction)
    gt_html = ground_truth_html(prediction)
    metric_full = TEDS(structure_only=False, n_jobs=1)
    metric_struct = TEDS(structure_only=True, n_jobs=1)
    return (
        pred_path_text,
        float(metric_full.evaluate(pred_html, gt_html)),
        float(metric_struct.evaluate(pred_html, gt_html)),
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    per_table_path = run_dir / "per_table.json"
    summary_path = run_dir / "summary.json"

    per_table = json.loads(per_table_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    pred_paths = [str(Path(row["pred_path"])) for row in per_table]
    if args.jobs == 1:
        scored = [score_prediction(path) for path in pred_paths]
    else:
        with mp.Pool(processes=args.jobs) as pool:
            scored = pool.map(score_prediction, pred_paths)

    score_by_path = {path: (teds, teds_s) for path, teds, teds_s in scored}
    rescored_rows: list[dict[str, Any]] = []
    for row in per_table:
        pred_path = str(Path(row["pred_path"]))
        teds, teds_s = score_by_path[pred_path]
        rescored_rows.append({**row, "teds": teds, "teds_s": teds_s})

        if args.write_predictions:
            prediction = json.loads(Path(pred_path).read_text(encoding="utf-8"))
            prediction["evaluation_result"] = {"teds": teds, "teds_s": teds_s}
            Path(pred_path).write_text(
                json.dumps(prediction, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    teds_values = [row["teds"] for row in rescored_rows]
    teds_s_values = [row["teds_s"] for row in rescored_rows]

    summary["mean_teds"] = statistics.mean(teds_values) if teds_values else None
    summary["mean_teds_s"] = statistics.mean(teds_s_values) if teds_s_values else None
    summary["paper_metric_rescored"] = True
    summary["paper_metric_note"] = (
        "Recomputed with PubTabNet TEDS semantics: TEDS() and "
        "TEDS(structure_only=True), without ignored nodes or cell-content stripping."
    )

    per_table_path.write_text(
        json.dumps(rescored_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    perfect_teds = sum(1 for value in teds_values if value == 1.0)
    perfect_teds_s = sum(1 for value in teds_s_values if value == 1.0)
    perfect_both = sum(
        1
        for teds, teds_s in zip(teds_values, teds_s_values)
        if teds == 1.0 and teds_s == 1.0
    )
    print(f"Run: {run_dir}")
    print(f"Rows: {len(rescored_rows)}")
    print(f"Mean TEDS: {summary['mean_teds']}")
    print(f"Mean TEDS-S: {summary['mean_teds_s']}")
    print(f"TEDS=1: {perfect_teds}")
    print(f"TEDS-S=1: {perfect_teds_s}")
    print(f"Both=1: {perfect_both}")


if __name__ == "__main__":
    main()
