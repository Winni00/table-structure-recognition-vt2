#!/cluster/home/trinhwin/vt2/docling/.venv/bin/python
from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
PUBTABNET_JSONL = ROOT / "data/pubtabnet_hf/extracted/pubtabnet/PubTabNet_2.0.0.jsonl"
PUBTABNET_IMAGES = ROOT / "data/pubtabnet_hf/extracted/pubtabnet/val"
BASE_EXPERIMENT = ROOT / "tableformer_pubtabnet_repro_bbox_experiment.py"
OUT_DIR = ROOT / "results/tableformer_pubtabnet_hf_bbox_reconstruction/line_v4_preprocessing"

OVERLAP_THRESHOLD = 0.10


REFERENCE_FILENAMES = [
    "PMC4816782_006_00.png",
    "PMC4788677_006_00.png",
    "PMC5066074_008_00.png",
    "PMC3564928_006_00.png",
    "PMC5828203_005_00.png",
]


def load_base_module():
    spec = importlib.util.spec_from_file_location("tf_bbox_exp", BASE_EXPERIMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


EXP = load_base_module()


def load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def load_val_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with PUBTABNET_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") == "val":
                rows.append(row)
    return rows


def bbox_from_cell(cell: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = cell.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return None


def bbox_area(bbox: tuple[float, float, float, float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def visible_text(cell: dict[str, Any]) -> str:
    return EXP.visible_cell_text(EXP.rich_text_from_tokens(cell.get("tokens", [])))


def crop_from_boxes(
    boxes: list[tuple[float, float, float, float] | None],
    image_size: tuple[int, int],
    margin: int = 10,
) -> tuple[int, int, int, int]:
    valid = [bbox for bbox in boxes if bbox is not None and bbox_area(bbox) > 0]
    if not valid:
        return 0, 0, image_size[0], image_size[1]
    return (
        max(0, math.floor(min(bbox[0] for bbox in valid) - margin)),
        max(0, math.floor(min(bbox[1] for bbox in valid) - margin)),
        min(image_size[0], math.ceil(max(bbox[2] for bbox in valid) + margin)),
        min(image_size[1], math.ceil(max(bbox[3] for bbox in valid) + margin)),
    )


def group_positions(mask: np.ndarray, min_gap: int = 2) -> list[tuple[int, int, int]]:
    """Group neighbouring true positions into (start, end, center)."""
    idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        return []
    groups: list[list[int]] = [[int(idxs[0])]]
    for value in idxs[1:]:
        if int(value) <= groups[-1][-1] + min_gap:
            groups[-1].append(int(value))
        else:
            groups.append([int(value)])
    return [(group[0], group[-1], int(round(statistics.median(group)))) for group in groups]


def detect_ruling_lines(
    image: Image.Image,
    crop: tuple[int, int, int, int],
) -> tuple[list[int], list[int]]:
    """Detect strong horizontal/vertical ruling lines inside a table crop.

    This is intentionally a diagnostic/reconstruction helper, not an OCR step.
    It only trusts long dark pixel runs, because text bboxes are too tight to
    infer empty-cell geometry reliably.
    """
    x0, y0, x1, y1 = crop
    cropped = image.crop(crop).convert("L")
    arr = np.array(cropped)
    dark = arr < 90

    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    # Long horizontal/vertical dark runs. Text has local density, table rules
    # usually stretch over a large share of the table crop.
    horizontal_density = dark.mean(axis=1)
    vertical_density = dark.mean(axis=0)
    horizontal_mask = horizontal_density > 0.35
    vertical_mask = vertical_density > 0.25

    horizontal = [center + y0 for start, end, center in group_positions(horizontal_mask) if end - start <= 8]
    vertical = [center + x0 for start, end, center in group_positions(vertical_mask) if end - start <= 8]

    # Keep only lines that are plausible table rules, not page/image borders.
    horizontal = [pos for pos in horizontal if y0 + 1 < pos < y1 - 1]
    vertical = [pos for pos in vertical if x0 + 1 < pos < x1 - 1]
    return horizontal, vertical


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def monotonic_lines_from_centers(
    centers: list[float | None],
    fallback_min: float,
    fallback_max: float,
) -> list[float]:
    n = len(centers)
    if n == 0:
        return [fallback_min, fallback_max]
    filled = list(centers)
    fallback_centers = [
        fallback_min + (idx + 0.5) * (fallback_max - fallback_min) / max(1, n)
        for idx in range(n)
    ]
    for idx, value in enumerate(filled):
        if value is not None:
            continue
        left = idx - 1
        while left >= 0 and filled[left] is None:
            left -= 1
        right = idx + 1
        while right < n and filled[right] is None:
            right += 1
        if left >= 0 and right < n and filled[left] is not None and filled[right] is not None:
            ratio = (idx - left) / (right - left)
            filled[idx] = float(filled[left]) + ratio * (float(filled[right]) - float(filled[left]))
        else:
            filled[idx] = fallback_centers[idx]

    numeric = [float(v) for v in filled]
    if any(numeric[idx] <= numeric[idx - 1] for idx in range(1, len(numeric))):
        numeric = fallback_centers
    lines = [fallback_min]
    lines.extend((left + right) / 2.0 for left, right in zip(numeric[:-1], numeric[1:]))
    lines.append(fallback_max)
    return lines


def snap_lines_to_rulings(
    guessed_lines: list[float],
    detected_lines: list[int],
    tolerance: float,
) -> tuple[list[float], int]:
    """Snap estimated boundaries to nearby visual ruling lines."""
    if not detected_lines:
        return guessed_lines, 0
    snapped: list[float] = []
    used = 0
    for line in guessed_lines:
        nearest = min(detected_lines, key=lambda value: abs(value - line))
        if abs(nearest - line) <= tolerance:
            snapped.append(float(nearest))
            used += 1
        else:
            snapped.append(float(line))
    # Preserve strict monotonicity.
    for idx in range(1, len(snapped)):
        if snapped[idx] <= snapped[idx - 1] + 1:
            return guessed_lines, 0
    return snapped, used


def reconstruct_missing_cell_bboxes_line_v4(
    row: dict[str, Any],
    image: Image.Image,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared = deepcopy(row)
    records, num_rows, num_cols = EXP.parse_pubtabnet_grid_cells(prepared)
    if records:
        num_rows = max(num_rows, max(record["end_row"] for record in records))
        num_cols = max(num_cols, max(record["end_col"] for record in records))

    existing_boxes = [bbox_from_cell(record["cell"]) for record in records]
    valid_boxes = [bbox for bbox in existing_boxes if bbox is not None and bbox_area(bbox) > 0]
    crop = crop_from_boxes(valid_boxes, image.size, margin=8)
    crop_x0, crop_y0, crop_x1, crop_y1 = crop
    horizontal_lines, vertical_lines = detect_ruling_lines(image, crop)

    col_centers: list[list[float]] = [[] for _ in range(num_cols)]
    row_centers: list[list[float]] = [[] for _ in range(num_rows)]
    for record in records:
        bbox = bbox_from_cell(record["cell"])
        if bbox is None or bbox_area(bbox) <= 0:
            continue
        x0, y0, x1, y1 = bbox
        if record["col_span"] == 1:
            col_centers[record["start_col"]].append((x0 + x1) / 2.0)
        if record["row_span"] == 1:
            row_centers[record["start_row"]].append((y0 + y1) / 2.0)

    x_lines = monotonic_lines_from_centers(
        [median_or_none(values) for values in col_centers],
        float(crop_x0),
        float(crop_x1),
    )
    y_lines = monotonic_lines_from_centers(
        [median_or_none(values) for values in row_centers],
        float(crop_y0),
        float(crop_y1),
    )
    x_lines, snapped_x = snap_lines_to_rulings(x_lines, vertical_lines, tolerance=max(6.0, (crop_x1 - crop_x0) * 0.025))
    y_lines, snapped_y = snap_lines_to_rulings(y_lines, horizontal_lines, tolerance=max(6.0, (crop_y1 - crop_y0) * 0.025))

    reconstructed = 0
    empty_reconstructed = 0
    for record in records:
        cell = record["cell"]
        cell["is_empty"] = not visible_text(cell)
        if bbox_from_cell(cell) is not None:
            cell["bbox_reconstructed"] = False
            continue
        x0 = x_lines[record["start_col"]]
        x1 = x_lines[record["end_col"]]
        y0 = y_lines[record["start_row"]]
        y1 = y_lines[record["end_row"]]
        cell["bbox"] = [
            max(0.0, min(float(image.width), x0)),
            max(0.0, min(float(image.height), y0)),
            max(0.0, min(float(image.width), x1)),
            max(0.0, min(float(image.height), y1)),
        ]
        cell["bbox_reconstructed"] = True
        reconstructed += 1
        if cell["is_empty"]:
            empty_reconstructed += 1

    return prepared, {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "total_cells": len(records),
        "reconstructed_bboxes": reconstructed,
        "reconstructed_empty_bboxes": empty_reconstructed,
        "detected_horizontal_lines": len(horizontal_lines),
        "detected_vertical_lines": len(vertical_lines),
        "snapped_x_lines": snapped_x,
        "snapped_y_lines": snapped_y,
        "crop": [crop_x0, crop_y0, crop_x1, crop_y1],
    }


def grid_disjoint(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a["end_row"] <= b["start_row"]
        or b["end_row"] <= a["start_row"]
        or a["end_col"] <= b["start_col"]
        or b["end_col"] <= a["start_col"]
    )


def inter_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def diagnose(row: dict[str, Any]) -> dict[str, Any]:
    image_path = PUBTABNET_IMAGES / row["filename"]
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        prepared, stats = reconstruct_missing_cell_bboxes_line_v4(row, image)
    records, _, _ = EXP.parse_pubtabnet_grid_cells(prepared)
    reconstructed = [record for record in records if record["cell"].get("bbox_reconstructed")]
    missing_after = [record for record in records if bbox_from_cell(record["cell"]) is None]
    invalid = [
        record
        for record in reconstructed
        if bbox_from_cell(record["cell"]) is None or bbox_area(bbox_from_cell(record["cell"])) <= 0
    ]
    overlaps = []
    for record in reconstructed:
        record_bbox = bbox_from_cell(record["cell"])
        if record_bbox is None or bbox_area(record_bbox) <= 0:
            continue
        for other in records:
            if other is record or not grid_disjoint(record, other):
                continue
            other_bbox = bbox_from_cell(other["cell"])
            if other_bbox is None or bbox_area(other_bbox) <= 0:
                continue
            overlap = inter_area(record_bbox, other_bbox)
            denom = min(bbox_area(record_bbox), bbox_area(other_bbox))
            if denom > 0 and overlap / denom > OVERLAP_THRESHOLD:
                overlaps.append((record, other, overlap / denom))
    return {
        "filename": row["filename"],
        "grid": f"{stats['num_rows']}x{stats['num_cols']}",
        "reconstructed_bboxes": len(reconstructed),
        "missing_after_reconstruction": len(missing_after),
        "invalid_reconstructed_bboxes": len(invalid),
        "overlap_flags": len(overlaps),
        "detected_horizontal_lines": stats["detected_horizontal_lines"],
        "detected_vertical_lines": stats["detected_vertical_lines"],
        "snapped_x_lines": stats["snapped_x_lines"],
        "snapped_y_lines": stats["snapped_y_lines"],
    }


def draw_overlay(
    image: Image.Image,
    crop: tuple[int, int, int, int],
    title: str,
    blue: list[tuple[float, float, float, float] | None],
    orange: list[tuple[float, float, float, float] | None],
    horizontal_lines: list[int] | None = None,
    vertical_lines: list[int] | None = None,
) -> Image.Image:
    cropped = image.crop(crop).convert("RGB")
    scale = min(2.8, max(1.0, 850 / max(cropped.width, cropped.height)))
    if scale > 1.01:
        cropped = cropped.resize((int(cropped.width * scale), int(cropped.height * scale)))
    draw = ImageDraw.Draw(cropped)
    ox, oy = crop[0], crop[1]

    def transform(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return (
            (bbox[0] - ox) * scale,
            (bbox[1] - oy) * scale,
            (bbox[2] - ox) * scale,
            (bbox[3] - oy) * scale,
        )

    for y in horizontal_lines or []:
        draw.line([(0, (y - oy) * scale), (cropped.width, (y - oy) * scale)], fill="#cc00cc", width=1)
    for x in vertical_lines or []:
        draw.line([((x - ox) * scale, 0), ((x - ox) * scale, cropped.height)], fill="#cc00cc", width=1)
    for bbox in blue:
        if bbox is not None and bbox_area(bbox) > 0:
            draw.rectangle(transform(bbox), outline="#1f77b4", width=2)
    for bbox in orange:
        if bbox is not None and bbox_area(bbox) > 0:
            draw.rectangle(transform(bbox), outline="#ff8c00", width=3)

    panel = Image.new("RGB", (cropped.width, cropped.height + 58), "white")
    panel.paste(cropped, (0, 58))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text((8, 8), title, font=load_font(17), fill="black")
    panel_draw.text(
        (8, 32),
        "blue=original bbox, orange=line-v4 reconstructed empty bbox, magenta=detected ruling lines",
        font=load_font(11),
        fill="gray",
    )
    return panel


def draw_reference_sample(row: dict[str, Any]) -> None:
    image = Image.open(PUBTABNET_IMAGES / row["filename"]).convert("RGB")
    prepared, stats = reconstruct_missing_cell_bboxes_line_v4(row, image)
    records, _, _ = EXP.parse_pubtabnet_grid_cells(prepared)
    original_boxes = [bbox_from_cell(cell) for cell in row["html"]["cells"] if bbox_from_cell(cell)]
    all_boxes = [bbox_from_cell(record["cell"]) for record in records if bbox_from_cell(record["cell"])]
    reconstructed_boxes = [
        bbox_from_cell(record["cell"])
        for record in records
        if record["cell"].get("bbox_reconstructed") and bbox_from_cell(record["cell"])
    ]
    crop = tuple(stats["crop"])
    horizontal_lines, vertical_lines = detect_ruling_lines(image, crop)
    left = draw_overlay(image, crop, "Original PubTabNet GT cell boxes", original_boxes, [])
    right = draw_overlay(
        image,
        crop,
        f"Line-v4 reconstructed GT cell boxes ({stats['reconstructed_empty_bboxes']})",
        [bbox for bbox in all_boxes if bbox not in reconstructed_boxes],
        reconstructed_boxes,
        horizontal_lines,
        vertical_lines,
    )
    canvas = Image.new("RGB", (left.width + right.width + 24, max(left.height, right.height) + 72), "white")
    draw = ImageDraw.Draw(canvas)
    stem = Path(row["filename"]).stem
    draw.text((8, 8), stem, font=load_font(24), fill="black")
    draw.text(
        (8, 38),
        (
            f"GT grid stays {stats['num_rows']}x{stats['num_cols']}; "
            f"reconstructed_empty_bboxes={stats['reconstructed_empty_bboxes']}; "
            f"snapped y/x={stats['snapped_y_lines']}/{stats['snapped_x_lines']}"
        ),
        font=load_font(14),
        fill="black",
    )
    canvas.paste(left, (0, 72))
    canvas.paste(right, (left.width + 24, 72))
    out = OUT_DIR / "sample_visuals" / f"{stem}_line_v4_gt_boxes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_val_rows()
    diagnostics = [diagnose(row) for row in rows]
    with (OUT_DIR / "per_table_diagnostic.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics)
    summary = {
        "tables": len(diagnostics),
        "tables_with_reconstruction": sum(1 for row in diagnostics if row["reconstructed_bboxes"] > 0),
        "reconstructed_bboxes": sum(row["reconstructed_bboxes"] for row in diagnostics),
        "missing_after_reconstruction": sum(row["missing_after_reconstruction"] for row in diagnostics),
        "tables_with_invalid_reconstructed_bboxes": sum(1 for row in diagnostics if row["invalid_reconstructed_bboxes"] > 0),
        "invalid_reconstructed_bboxes": sum(row["invalid_reconstructed_bboxes"] for row in diagnostics),
        "tables_with_overlap_flags": sum(1 for row in diagnostics if row["overlap_flags"] > 0),
        "overlap_flags": sum(row["overlap_flags"] for row in diagnostics),
        "detected_horizontal_lines_total": sum(row["detected_horizontal_lines"] for row in diagnostics),
        "detected_vertical_lines_total": sum(row["detected_vertical_lines"] for row in diagnostics),
        "snapped_x_lines_total": sum(row["snapped_x_lines"] for row in diagnostics),
        "snapped_y_lines_total": sum(row["snapped_y_lines"] for row in diagnostics),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    by_filename = {row["filename"]: row for row in rows}
    for filename in REFERENCE_FILENAMES:
        if filename in by_filename:
            draw_reference_sample(by_filename[filename])
    print(json.dumps(summary, indent=2))
    print(f"Wrote line-v4 preprocessing diagnostic to {OUT_DIR}")


if __name__ == "__main__":
    main()
