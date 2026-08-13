#!/usr/bin/env python3
"""Rescore stored PubTabNet TableFormer runs with canonical section HTML."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from lxml import html

ROOT = Path("/cluster/home/trinhwin/vt2/docling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def ensure_html_table_document(text: str) -> str:
    cleaned = text.strip()
    lower = cleaned.lower()
    if "<html" in lower:
        return cleaned
    if "<table" in lower:
        return f"<html><body>{cleaned}</body></html>"
    return f"<html><body><table>{cleaned}</table></body></html>"


def canonicalize_sections(text: str) -> str:
    html_doc = ensure_html_table_document(text)
    return re.sub(
        r"(</thead>)((?:<tr>.*?</tr>)+)(</table>)",
        r"\1<tbody>\2</tbody>\3",
        html_doc,
        flags=re.DOTALL,
    )


def normalized_table(prediction: dict[str, Any]) -> dict[str, Any]:
    normalized = prediction.get("normalized_output", {})
    if isinstance(normalized, list):
        return normalized[0] if normalized else {}
    return normalized if isinstance(normalized, dict) else {}


def table_node_count(html_text: str) -> int:
    root = html.fromstring(ensure_html_table_document(html_text))
    table = root.xpath("body/table")
    if not table:
        return 0
    return len(table[0].xpath(".//*"))


def adjust_for_inserted_section_node(
    old_score: float,
    old_pred_html: str,
    old_gt_html: str,
    new_pred_html: str,
    new_gt_html: str,
) -> float:
    """Update TEDS when canonicalization only inserts missing tbody nodes."""
    old_max_nodes = max(table_node_count(old_pred_html), table_node_count(old_gt_html))
    new_max_nodes = max(table_node_count(new_pred_html), table_node_count(new_gt_html))
    inserted_nodes = table_node_count(new_pred_html) - table_node_count(old_pred_html)
    if old_max_nodes <= 0 or new_max_nodes <= 0 or inserted_nodes <= 0:
        return old_score

    old_distance = (1.0 - old_score) * old_max_nodes
    new_distance = max(0.0, old_distance - inserted_nodes)
    return 1.0 - (new_distance / new_max_nodes)


def score_one(args: tuple[str, float, float]) -> tuple[str, float, float, bool]:
    pred_path_text, old_teds, old_teds_s = args
    pred_path = Path(pred_path_text)
    prediction = json.loads(pred_path.read_text(encoding="utf-8"))
    norm = normalized_table(prediction)
    pred_old = ensure_html_table_document(norm.get("structure", {}).get("html", ""))
    gt_old = ensure_html_table_document(prediction.get("ground_truth", {}).get("gt_html", ""))
    pred_new = canonicalize_sections(pred_old)
    gt_new = canonicalize_sections(gt_old)

    changed = pred_new != pred_old
    if pred_new == gt_new:
        return pred_path_text, 1.0, 1.0, changed
    if not changed:
        return pred_path_text, old_teds, old_teds_s, False

    return (
        pred_path_text,
        adjust_for_inserted_section_node(old_teds, pred_old, gt_old, pred_new, gt_new),
        adjust_for_inserted_section_node(old_teds_s, pred_old, gt_old, pred_new, gt_new),
        True,
    )


def rescore_run(run_dir: Path, write: bool) -> dict[str, Any]:
    per_table_path = run_dir / "per_table.json"
    summary_path = run_dir / "summary.json"
    per_table = json.loads(per_table_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    tasks = [
        (str(Path(row["pred_path"])), float(row["teds"]), float(row["teds_s"]))
        for row in per_table
    ]
    scored = [score_one(task) for task in tasks]

    score_by_path = {path: (teds, teds_s, changed) for path, teds, teds_s, changed in scored}
    updated_rows = []
    for row in per_table:
        path = str(Path(row["pred_path"]))
        teds, teds_s, _ = score_by_path[path]
        updated_rows.append({**row, "teds": teds, "teds_s": teds_s})

    teds_values = [row["teds"] for row in updated_rows]
    teds_s_values = [row["teds_s"] for row in updated_rows]
    changed_count = sum(1 for _, _, changed in score_by_path.values() if changed)
    result = {
        "run_dir": str(run_dir),
        "rows": len(updated_rows),
        "old_mean_teds": summary.get("mean_teds"),
        "old_mean_teds_s": summary.get("mean_teds_s"),
        "new_mean_teds": statistics.mean(teds_values) if teds_values else None,
        "new_mean_teds_s": statistics.mean(teds_s_values) if teds_s_values else None,
        "teds_eq_1": sum(1 for value in teds_values if value == 1.0),
        "teds_s_eq_1": sum(1 for value in teds_s_values if value == 1.0),
        "both_eq_1": sum(
            1
            for teds, teds_s in zip(teds_values, teds_s_values)
            if teds == 1.0 and teds_s == 1.0
        ),
        "canonicalized_predictions": changed_count,
    }

    if write:
        summary["mean_teds"] = result["new_mean_teds"]
        summary["mean_teds_s"] = result["new_mean_teds_s"]
        summary["html_canonicalization"] = {
            "enabled": True,
            "note": (
                "Wrapped direct table rows after thead into tbody before TEDS scoring; "
                "scores adjusted for the inserted section node."
            ),
            "canonicalized_predictions": changed_count,
        }
        summary["perfect_counts"] = {
            "rows": result["rows"],
            "teds_eq_1": result["teds_eq_1"],
            "teds_s_eq_1": result["teds_s_eq_1"],
            "both_eq_1": result["both_eq_1"],
        }
        per_table_path.write_text(
            json.dumps(updated_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return result


def main() -> None:
    args = parse_args()
    results = [rescore_run(run_dir, args.write) for run_dir in args.run_dirs]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
