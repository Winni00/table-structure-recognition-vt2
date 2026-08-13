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

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
PUBTABNET_JSONL = ROOT / "data/pubtabnet_hf/extracted/pubtabnet/PubTabNet_2.0.0.jsonl"
PUBTABNET_IMAGES = ROOT / "data/pubtabnet_hf/extracted/pubtabnet/val"
BASE_EXPERIMENT = ROOT / "tableformer_pubtabnet_repro_bbox_experiment.py"
OUT_DIR = ROOT / "results/tableformer_pubtabnet_hf_bbox_reconstruction/local_v3_preprocessing"

OVERLAP_THRESHOLD = 0.10


def load_base_module():
    spec = importlib.util.spec_from_file_location("tf_bbox_exp", BASE_EXPERIMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


EXP = load_base_module()


def bbox_from_cell(cell: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = cell.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return None


def bbox_area(bbox: tuple[float, float, float, float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def fill_grid_lines(candidates: list[list[float]], fallback_min: float, fallback_max: float) -> list[float]:
    n = len(candidates) - 1
    known: list[float | None] = [statistics.median(values) if values else None for values in candidates]
    if n <= 0:
        return [fallback_min, fallback_max]
    if known[0] is None:
        known[0] = fallback_min
    if known[n] is None:
        known[n] = fallback_max
    for idx, value in enumerate(known):
        if value is not None:
            continue
        left = idx - 1
        while left >= 0 and known[left] is None:
            left -= 1
        right = idx + 1
        while right <= n and known[right] is None:
            right += 1
        if left >= 0 and right <= n and known[left] is not None and known[right] is not None:
            ratio = (idx - left) / (right - left)
            known[idx] = float(known[left]) + ratio * (float(known[right]) - float(known[left]))
        else:
            known[idx] = fallback_min + (fallback_max - fallback_min) * idx / n
    lines = [float(value) for value in known]
    min_step = max(1e-3, (fallback_max - fallback_min) / max(1, n) * 0.01)
    if any(lines[idx] <= lines[idx - 1] + min_step for idx in range(1, len(lines))):
        # Safer fallback than zero-width or non-monotonic boundaries.
        return [fallback_min + idx * (fallback_max - fallback_min) / n for idx in range(n + 1)]
    return lines


def reconstruct_missing_cell_bboxes_local_v3(
    row: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct missing PubTabNet cell boxes using local row bands.

    This variant follows the spirit of the TableFormer post-processing notes:
    row/column bands are estimated robustly, and missing cells are snapped to
    the local row band instead of a global center-interpolated grid. It avoids
    the v2 failure mode where an empty box can start far above its visual row.
    """
    prepared = deepcopy(row)
    records, num_rows, num_cols = EXP.parse_pubtabnet_grid_cells(prepared)
    if records:
        num_rows = max(num_rows, max(record["end_row"] for record in records))
        num_cols = max(num_cols, max(record["end_col"] for record in records))

    existing_boxes = [bbox_from_cell(record["cell"]) for record in records]
    existing_boxes = [bbox for bbox in existing_boxes if bbox is not None and bbox_area(bbox) > 0]
    if existing_boxes:
        table_x0 = max(0.0, min(bbox[0] for bbox in existing_boxes))
        table_y0 = max(0.0, min(bbox[1] for bbox in existing_boxes))
        table_x1 = min(float(image_width), max(bbox[2] for bbox in existing_boxes))
        table_y1 = min(float(image_height), max(bbox[3] for bbox in existing_boxes))
    else:
        table_x0, table_y0, table_x1, table_y1 = 0.0, 0.0, float(image_width), float(image_height)

    x_candidates = [[] for _ in range(num_cols + 1)]
    y_candidates = [[] for _ in range(num_rows + 1)]
    row_bands: list[list[tuple[float, float]]] = [[] for _ in range(num_rows)]

    for record in records:
        bbox = bbox_from_cell(record["cell"])
        if bbox is None or bbox_area(bbox) <= 0:
            continue
        x0, y0, x1, y1 = bbox
        x_candidates[record["start_col"]].append(x0)
        x_candidates[record["end_col"]].append(x1)
        y_candidates[record["start_row"]].append(y0)
        y_candidates[record["end_row"]].append(y1)
        if record["row_span"] == 1 and record["start_row"] < num_rows:
            row_bands[record["start_row"]].append((y0, y1))

    x_lines = fill_grid_lines(x_candidates, table_x0, table_x1)
    y_lines = fill_grid_lines(y_candidates, table_y0, table_y1)

    reconstructed = 0
    empty_reconstructed = 0
    local_row_band_used = 0
    for record in records:
        cell = record["cell"]
        text = EXP.visible_cell_text(EXP.rich_text_from_tokens(cell.get("tokens", [])))
        cell["is_empty"] = not text
        if bbox_from_cell(cell) is not None:
            cell["bbox_reconstructed"] = False
            continue

        x0 = x_lines[record["start_col"]]
        x1 = x_lines[record["end_col"]]

        # Local row band: use the visible cells in the same logical row. This is
        # the main v3 change and mirrors the paper's row-band reasoning.
        if record["row_span"] == 1 and record["start_row"] < len(row_bands) and row_bands[record["start_row"]]:
            y0 = min(pair[0] for pair in row_bands[record["start_row"]])
            y1 = max(pair[1] for pair in row_bands[record["start_row"]])
            local_row_band_used += 1
        else:
            y0 = y_lines[record["start_row"]]
            y1 = y_lines[record["end_row"]]

        if x1 <= x0:
            width = max(1.0, (table_x1 - table_x0) / max(1, num_cols))
            x1 = min(float(image_width), x0 + width)
        if y1 <= y0:
            height = max(1.0, (table_y1 - table_y0) / max(1, num_rows))
            y1 = min(float(image_height), y0 + height)

        cell["bbox"] = [
            max(0.0, min(float(image_width), x0)),
            max(0.0, min(float(image_height), y0)),
            max(0.0, min(float(image_width), x1)),
            max(0.0, min(float(image_height), y1)),
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
        "local_row_band_used": local_row_band_used,
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
    with Image.open(PUBTABNET_IMAGES / row["filename"]) as image:
        width, height = image.size
    prepared, stats = reconstruct_missing_cell_bboxes_local_v3(row, width, height)
    records, num_rows, num_cols = EXP.parse_pubtabnet_grid_cells(prepared)
    if records:
        num_rows = max(num_rows, max(record["end_row"] for record in records))
        num_cols = max(num_cols, max(record["end_col"] for record in records))
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
        "grid": f"{num_rows}x{num_cols}",
        "reconstructed_bboxes": len(reconstructed),
        "missing_after_reconstruction": len(missing_after),
        "invalid_reconstructed_bboxes": len(invalid),
        "overlap_flags": len(overlaps),
        "local_row_band_used": stats["local_row_band_used"],
    }


def load_val_rows() -> list[dict[str, Any]]:
    with PUBTABNET_JSONL.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if json.loads(line).get("split") == "val"]


def load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def crop_from_boxes(
    boxes: list[tuple[float, float, float, float] | None],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    valid = [bbox for bbox in boxes if bbox is not None and bbox_area(bbox) > 0]
    if not valid:
        return 0, 0, image_size[0], image_size[1]
    x0 = max(0, math.floor(min(bbox[0] for bbox in valid) - 20))
    y0 = max(0, math.floor(min(bbox[1] for bbox in valid) - 20))
    x1 = min(image_size[0], math.ceil(max(bbox[2] for bbox in valid) + 20))
    y1 = min(image_size[1], math.ceil(max(bbox[3] for bbox in valid) + 20))
    return x0, y0, x1, y1


def draw_overlay(
    image: Image.Image,
    crop: tuple[int, int, int, int],
    title: str,
    blue: list[tuple[float, float, float, float] | None],
    orange: list[tuple[float, float, float, float] | None],
) -> Image.Image:
    cropped = image.crop(crop).convert("RGB")
    scale = min(3.0, max(1.0, 850 / max(cropped.width, cropped.height)))
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

    for bbox in blue:
        if bbox is not None and bbox_area(bbox) > 0:
            draw.rectangle(transform(bbox), outline="#1f77b4", width=2)
    for bbox in orange:
        if bbox is not None and bbox_area(bbox) > 0:
            draw.rectangle(transform(bbox), outline="#ff8c00", width=3)
    panel = Image.new("RGB", (cropped.width, cropped.height + 54), "white")
    panel.paste(cropped, (0, 54))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text((8, 8), title, font=load_font(17), fill="black")
    panel_draw.text((8, 32), "blue=original bbox, orange=reconstructed empty bbox", font=load_font(12), fill="gray")
    return panel


def draw_reference_sample(filename: str) -> None:
    row = next(row for row in load_val_rows() if row["filename"] == filename)
    image = Image.open(PUBTABNET_IMAGES / filename).convert("RGB")
    prepared, stats = reconstruct_missing_cell_bboxes_local_v3(row, image.width, image.height)
    original_boxes = [bbox_from_cell(cell) for cell in row["html"]["cells"] if bbox_from_cell(cell)]
    all_boxes = [bbox_from_cell(cell) for cell in prepared["html"]["cells"] if bbox_from_cell(cell)]
    reconstructed_boxes = [
        bbox_from_cell(cell)
        for cell in prepared["html"]["cells"]
        if cell.get("bbox_reconstructed") and bbox_from_cell(cell)
    ]
    crop = crop_from_boxes(all_boxes or original_boxes, image.size)
    left = draw_overlay(image, crop, "Original PubTabNet GT boxes", original_boxes, [])
    right = draw_overlay(
        image,
        crop,
        f"Local-v3 reconstructed boxes ({stats['reconstructed_empty_bboxes']})",
        [bbox for bbox in all_boxes if bbox not in reconstructed_boxes],
        reconstructed_boxes,
    )
    canvas = Image.new("RGB", (left.width + right.width + 24, max(left.height, right.height) + 70), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), filename.replace(".png", ""), font=load_font(24), fill="black")
    draw.text((8, 38), f"Local-v3 preprocessing; GT stays {stats['num_rows']}x{stats['num_cols']}", font=load_font(15), fill="black")
    canvas.paste(left, (0, 70))
    canvas.paste(right, (left.width + 24, 70))
    out = OUT_DIR / "visual_examples" / f"{filename.replace('.png', '')}_local_v3.png"
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
        "local_row_band_used": sum(row["local_row_band_used"] for row in diagnostics),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_reference_sample("PMC4816782_006_00.png")
    print(json.dumps(summary, indent=2))
    print(f"Wrote local-v3 preprocessing diagnostic to {OUT_DIR}")


if __name__ == "__main__":
    main()
