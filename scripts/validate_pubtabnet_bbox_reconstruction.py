#!/cluster/home/trinhwin/vt2/docling/.venv/bin/python
from __future__ import annotations

import csv
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
PUBTABNET_JSONL = ROOT / "data/pubtabnet_hf/extracted/pubtabnet/PubTabNet_2.0.0.jsonl"
PUBTABNET_IMAGES = ROOT / "data/pubtabnet_hf/extracted/pubtabnet/val"
EXPERIMENT_SCRIPT = ROOT / "tableformer_pubtabnet_repro_bbox_experiment.py"
OUT_DIR = ROOT / "results/tableformer_pubtabnet_hf_bbox_reconstruction/preprocessing_diagnostic"

OVERLAP_THRESHOLD = 0.10
HUGE_AREA_RATIO = 8.0


def load_experiment_module():
    spec = importlib.util.spec_from_file_location("tf_bbox_exp", EXPERIMENT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


EXP = load_experiment_module()


def bbox_from_cell(cell: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = cell.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return None


def area(bbox: tuple[float, float, float, float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def inter_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def grid_disjoint(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a["end_row"] <= b["start_row"]
        or b["end_row"] <= a["start_row"]
        or a["end_col"] <= b["start_col"]
        or b["end_col"] <= a["start_col"]
    )


def visible_text(cell: dict[str, Any]) -> str:
    return EXP.visible_cell_text(EXP.rich_text_from_tokens(cell.get("tokens", [])))


def load_val_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with PUBTABNET_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") == "val":
                rows.append(row)
    return rows


def diagnose_row(row: dict[str, Any]) -> dict[str, Any]:
    image_path = PUBTABNET_IMAGES / row["filename"]
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    original_records, original_rows, original_cols = EXP.parse_pubtabnet_grid_cells(row)
    if original_records:
        original_rows = max(original_rows, max(record["end_row"] for record in original_records))
        original_cols = max(original_cols, max(record["end_col"] for record in original_records))

    original_missing = [
        record
        for record in original_records
        if not bbox_from_cell(record["cell"])
    ]

    prepared, stats = EXP.reconstruct_missing_cell_bboxes(row, image_width, image_height)
    records, num_rows, num_cols = EXP.parse_pubtabnet_grid_cells(prepared)
    if records:
        num_rows = max(num_rows, max(record["end_row"] for record in records))
        num_cols = max(num_cols, max(record["end_col"] for record in records))

    reconstructed = [
        record
        for record in records
        if record["cell"].get("bbox_reconstructed")
    ]
    missing_after = [
        record
        for record in records
        if not bbox_from_cell(record["cell"])
    ]

    invalid = []
    out_of_image = []
    for record in reconstructed:
        bbox = bbox_from_cell(record["cell"])
        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            invalid.append(record)
            continue
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > image_width or bbox[3] > image_height:
            out_of_image.append(record)

    original_areas = [
        area(bbox_from_cell(record["cell"]))
        for record in records
        if bbox_from_cell(record["cell"]) is not None
        and not record["cell"].get("bbox_reconstructed")
        and area(bbox_from_cell(record["cell"])) > 0
    ]
    median_original_area = statistics.median(original_areas) if original_areas else 0.0
    huge = []
    if median_original_area > 0:
        for record in reconstructed:
            bbox = bbox_from_cell(record["cell"])
            if area(bbox) > HUGE_AREA_RATIO * median_original_area:
                huge.append(record)

    overlaps = []
    for record in reconstructed:
        record_bbox = bbox_from_cell(record["cell"])
        record_area = area(record_bbox)
        if record_bbox is None or record_area <= 0:
            continue
        for other in records:
            if other is record or not grid_disjoint(record, other):
                continue
            other_bbox = bbox_from_cell(other["cell"])
            other_area = area(other_bbox)
            if other_bbox is None or other_area <= 0:
                continue
            overlap = inter_area(record_bbox, other_bbox)
            denom = min(record_area, other_area)
            if denom > 0 and overlap / denom > OVERLAP_THRESHOLD:
                overlaps.append(
                    {
                        "reconstructed_cell": [record["start_row"], record["start_col"]],
                        "other_cell": [other["start_row"], other["start_col"]],
                        "ratio": round(overlap / denom, 4),
                    }
                )

    zero_width = []
    zero_height = []
    for record in reconstructed:
        bbox = bbox_from_cell(record["cell"])
        if bbox is None:
            continue
        if bbox[2] <= bbox[0]:
            zero_width.append(record)
        if bbox[3] <= bbox[1]:
            zero_height.append(record)

    # A compact root-cause label helps compare variants without looking at all
    # individual boxes.
    causes = []
    if missing_after:
        causes.append("missing_after_reconstruction")
    if invalid:
        causes.append("invalid_zero_area")
    if overlaps:
        causes.append("overlap_from_tight_text_boundaries")
    if huge:
        causes.append("oversized_interpolated_cells")
    if not causes and reconstructed:
        causes.append("reconstructed_without_detected_geometry_issue")
    if not reconstructed:
        causes.append("no_reconstruction_needed")

    return {
        "filename": row["filename"],
        "grid": f"{num_rows}x{num_cols}",
        "num_rows": num_rows,
        "num_cols": num_cols,
        "total_cells": len(records),
        "original_missing_bboxes": len(original_missing),
        "reconstructed_bboxes": len(reconstructed),
        "missing_after_reconstruction": len(missing_after),
        "invalid_reconstructed_bboxes": len(invalid),
        "zero_width_bboxes": len(zero_width),
        "zero_height_bboxes": len(zero_height),
        "out_of_image_bboxes": len(out_of_image),
        "overlap_flags": len(overlaps),
        "huge_bbox_flags": len(huge),
        "median_original_bbox_area": round(median_original_area, 4),
        "causes": ";".join(causes),
        "example_overlaps": json.dumps(overlaps[:5]),
    }


def write_markdown_report(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    with_recon = sum(1 for row in rows if row["reconstructed_bboxes"] > 0)
    missing_after = sum(row["missing_after_reconstruction"] for row in rows)
    tables_missing_after = sum(1 for row in rows if row["missing_after_reconstruction"] > 0)
    invalid = sum(row["invalid_reconstructed_bboxes"] for row in rows)
    tables_invalid = sum(1 for row in rows if row["invalid_reconstructed_bboxes"] > 0)
    overlaps = sum(row["overlap_flags"] for row in rows)
    tables_overlap = sum(1 for row in rows if row["overlap_flags"] > 0)
    huge = sum(row["huge_bbox_flags"] for row in rows)
    tables_huge = sum(1 for row in rows if row["huge_bbox_flags"] > 0)

    def pct(value: int, denom: int = total) -> str:
        return f"{100 * value / denom:.2f}%" if denom else "0.00%"

    top_invalid = sorted(rows, key=lambda row: row["invalid_reconstructed_bboxes"], reverse=True)[:10]
    top_overlap = sorted(rows, key=lambda row: row["overlap_flags"], reverse=True)[:10]
    top_huge = sorted(rows, key=lambda row: row["huge_bbox_flags"], reverse=True)[:10]

    lines = [
        "# PubTabNet Empty-Cell BBox Reconstruction Diagnostic",
        "",
        "This report validates the preprocessing step before inference.",
        "",
        "## Summary",
        "",
        f"- Tables checked: {total}",
        f"- Tables requiring reconstruction: {with_recon} ({pct(with_recon)})",
        f"- Missing boxes after reconstruction: {missing_after} in {tables_missing_after} tables",
        f"- Invalid reconstructed boxes: {invalid} in {tables_invalid} tables ({pct(tables_invalid)})",
        f"- Suspicious overlap flags: {overlaps} in {tables_overlap} tables ({pct(tables_overlap)})",
        f"- Oversized reconstructed-box flags: {huge} in {tables_huge} tables ({pct(tables_huge)})",
        "",
        "## Interpretation",
        "",
        "- The reconstruction fills all missing PubTabNet cell boxes, so the problem is not missing cells after preprocessing.",
        "- The main problem is geometry: inferred grid lines are derived from tight text/cell boxes, not from real table ruling lines.",
        "- This creates zero-width boxes, oversized boxes, and overlaps with logically disjoint cells.",
        "- Therefore this public reconstruction is not equivalent to IBM's prepared TableFormer format.",
        "",
    ]

    for title, table_rows, key in [
        ("Top Invalid / Zero-Area Cases", top_invalid, "invalid_reconstructed_bboxes"),
        ("Top Overlap Cases", top_overlap, "overlap_flags"),
        ("Top Oversized-Box Cases", top_huge, "huge_bbox_flags"),
    ]:
        lines.extend([f"## {title}", "", "| filename | grid | reconstructed | issue count | causes |", "|---|---:|---:|---:|---|"])
        for row in table_rows:
            lines.append(
                f"| {row['filename']} | {row['grid']} | {row['reconstructed_bboxes']} | {row[key]} | {row['causes']} |"
            )
        lines.append("")

    (OUT_DIR / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [diagnose_row(row) for row in load_val_rows()]

    csv_path = OUT_DIR / "per_table_diagnostic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (OUT_DIR / "per_table_diagnostic.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown_report(rows)
    print(f"Wrote diagnostics to {OUT_DIR}")


if __name__ == "__main__":
    main()
