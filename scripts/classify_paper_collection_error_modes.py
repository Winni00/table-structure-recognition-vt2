"""Classify Paper Collection failures into coarse error modes.

The goal is not to replace manual inspection. It creates a first-pass grouping:

- rotation: a rotated variant clearly improves the low-score sample
- text_ocr_error: structure score is high but full TEDS is much lower
- gt_convention_mismatch: spans/row layout differ in a way that is often visual-equivalent
- structure_layout_error: remaining low-structure cases

This script is intentionally conservative and writes all per-sample details so
the groups can be manually checked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_BASE_RUN = ROOT / "results/tflop_paper_collection_updated_crops_ocr_style_public"
DEFAULT_ROTATION_RUN = ROOT / "results/tflop_paper_collection_updated_crops_rotation_broad_badcases_public"
DEFAULT_OUT_DIR = DEFAULT_BASE_RUN / "error_mode_classification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument("--rotation-run-dir", type=Path, default=DEFAULT_ROTATION_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-teds", type=float, default=0.70)
    parser.add_argument("--max-teds-s", type=float, default=0.80)
    parser.add_argument("--rotation-min-delta-teds", type=float, default=0.05)
    parser.add_argument("--rotation-min-delta-teds-s", type=float, default=0.05)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    rows = load_json(path)
    return {
        row[0]: {
            "teds_s": float(row[-2]),
            "teds": float(row[-1]),
        }
        for row in rows
    }


def table_stats(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    rows = soup.find_all("tr")
    cells = soup.find_all(["td", "th"])
    span_cells = []
    for cell in cells:
        rowspan = int(cell.get("rowspan") or 1)
        colspan = int(cell.get("colspan") or 1)
        if rowspan > 1 or colspan > 1:
            span_cells.append(
                {
                    "text": cell.get_text(" ", strip=True)[:120],
                    "rowspan": rowspan,
                    "colspan": colspan,
                }
            )
    return {
        "rows": len(rows),
        "cells": len(cells),
        "rowspan_cells": sum(1 for cell in span_cells if cell["rowspan"] > 1),
        "colspan_cells": sum(1 for cell in span_cells if cell["colspan"] > 1),
        "span_cells": len(span_cells),
        "span_examples": span_cells[:5],
    }


def load_rotation_help(rotation_run_dir: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    summary_path = rotation_run_dir / "rotation_summary_gt_canon.json"
    if not summary_path.exists():
        return {}
    data = load_json(summary_path)
    helped = {}
    for row in data.get("rows", []):
        delta_teds = row.get("delta_best_vs_rot0_teds") or 0.0
        delta_teds_s = row.get("delta_best_vs_rot0_teds_s") or 0.0
        if (
            row.get("best_rotation") != 0
            and (delta_teds >= args.rotation_min_delta_teds or delta_teds_s >= args.rotation_min_delta_teds_s)
        ):
            helped[row["base_filename"]] = row
    return helped


def classify_record(record: dict[str, Any], rotation_help: dict[str, dict[str, Any]]) -> str:
    name = record["filename"]
    if name in rotation_help:
        return "rotation"
    if record["teds_s"] >= 0.90 and record["teds"] < 0.75:
        return "text_ocr_error"

    pred = record["pred_stats"]
    gt = record["gt_stats"]
    span_gap = abs(pred["span_cells"] - gt["span_cells"])
    row_gap = abs(pred["rows"] - gt["rows"])
    cell_gap = abs(pred["cells"] - gt["cells"])
    if gt["span_cells"] > 0 and span_gap >= 2 and row_gap <= 4:
        return "gt_convention_mismatch"
    if row_gap <= 2 and cell_gap <= 8 and record["teds_s"] < 0.90:
        return "gt_convention_mismatch"
    return "structure_layout_error"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    score_path = args.base_run_dir / "gt_canonicalized_rescore" / "ted_score_output.json"
    scores = load_scores(score_path)
    inference = load_json(args.base_run_dir / "full_model_inference.json")
    rotation_help = load_rotation_help(args.rotation_run_dir, args)

    records = []
    for filename, score in sorted(scores.items()):
        if not (score["teds"] < args.max_teds or score["teds_s"] < args.max_teds_s):
            continue
        item = inference[filename]
        record = {
            "filename": filename,
            **score,
            "pred_stats": table_stats(item["pred_string"]),
            "gt_stats": table_stats(item["answer_string"]),
            "rotation": rotation_help.get(filename),
        }
        record["class"] = classify_record(record, rotation_help)
        records.append(record)

    counts: dict[str, int] = {}
    for record in records:
        counts[record["class"]] = counts.get(record["class"], 0) + 1

    by_class: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_class.setdefault(record["class"], []).append(record)
    for values in by_class.values():
        values.sort(key=lambda row: (row["teds"], row["teds_s"]))

    output = {
        "base_run_dir": str(args.base_run_dir),
        "rotation_run_dir": str(args.rotation_run_dir),
        "score_path": str(score_path),
        "thresholds": {"max_teds": args.max_teds, "max_teds_s": args.max_teds_s},
        "low_score_samples": len(records),
        "counts": counts,
        "records": records,
    }
    (args.output_dir / "records.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Paper Collection Error Mode Classification",
        "",
        "This is a first-pass automatic grouping of low-score samples. It should be checked visually.",
        "",
        "## Summary",
        "",
        f"- Low-score samples: `{len(records)}`",
        f"- Rotation-helped samples loaded from broad rotation run: `{len(rotation_help)}`",
        "",
        "| Class | Count | Meaning |",
        "|---|---:|---|",
    ]
    meaning = {
        "rotation": "A rotated crop improved the score clearly.",
        "text_ocr_error": "TEDS-S is high but full TEDS is low, so structure is mostly okay but text/content is bad.",
        "gt_convention_mismatch": "HTML conventions differ, e.g. rowspan/linebreak/merged-cell encoding.",
        "structure_layout_error": "Likely real layout/structure problem after excluding the above.",
    }
    for klass in ["rotation", "text_ocr_error", "gt_convention_mismatch", "structure_layout_error"]:
        lines.append(f"| `{klass}` | {counts.get(klass, 0)} | {meaning[klass]} |")

    for klass in ["rotation", "text_ocr_error", "gt_convention_mismatch", "structure_layout_error"]:
        lines.extend(["", f"## {klass}", "", "| File | TEDS-S | TEDS | Notes |", "|---|---:|---:|---|"])
        for record in by_class.get(klass, [])[:15]:
            notes = []
            if record["rotation"]:
                notes.append(
                    f"best rot {record['rotation']['best_rotation']}, "
                    f"+{record['rotation']['delta_best_vs_rot0_teds']:.3f} TEDS"
                )
            notes.append(
                f"rows pred/GT {record['pred_stats']['rows']}/{record['gt_stats']['rows']}, "
                f"spans pred/GT {record['pred_stats']['span_cells']}/{record['gt_stats']['span_cells']}"
            )
            lines.append(
                f"| `{record['filename']}` | {record['teds_s']:.4f} | {record['teds']:.4f} | "
                f"{'; '.join(notes)} |"
            )

    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
