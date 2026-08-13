#!/cluster/home/trinhwin/vt2/docling/.venv/bin/python
from __future__ import annotations

import csv
import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
PUBTABNET_JSONL = ROOT / "data/pubtabnet_hf/extracted/pubtabnet/PubTabNet_2.0.0.jsonl"
PUBTABNET_IMAGES = ROOT / "data/pubtabnet_hf/extracted/pubtabnet/val"
RECON_PER_TABLE = (
    ROOT
    / "results/tableformer_pubtabnet_hf_bbox_reconstruction/val_full_9115_reconstructed/per_table.json"
)
EXPERIMENT_SCRIPT = ROOT / "tableformer_pubtabnet_repro_bbox_experiment.py"
OUT_DIR = ROOT / "results/tableformer_examples/pubtabnet_gt_original_vs_reconstructed_bboxes"


def load_experiment_module():
    spec = importlib.util.spec_from_file_location("tf_bbox_exp", EXPERIMENT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


EXP = load_experiment_module()


def load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(24)
FONT_SUB = load_font(18)
FONT_SMALL = load_font(14)


def bbox_from_cell(cell: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = cell.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    if isinstance(bbox, dict):
        return float(bbox["l"]), float(bbox["t"]), float(bbox["r"]), float(bbox["b"])
    return None


def table_shape_from_gt_html(row: dict[str, Any]) -> tuple[int, int]:
    # Use the same GT reconstruction parser as the adapter where possible.
    records, num_rows, num_cols = EXP.parse_pubtabnet_grid_cells(row)
    if records:
        num_rows = max(num_rows, max(record["end_row"] for record in records))
        num_cols = max(num_cols, max(record["end_col"] for record in records))
    return num_rows, num_cols


def load_requested_rows(filenames: set[str]) -> dict[str, dict[str, Any]]:
    rows = {}
    with PUBTABNET_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            filename = row["filename"]
            if filename in filenames:
                rows[filename] = row
            if len(rows) == len(filenames):
                break
    return rows


def choose_samples() -> list[dict[str, Any]]:
    per_table = json.loads(RECON_PER_TABLE.read_text(encoding="utf-8"))
    candidates = []
    # First pass: choose reasonably small tables with visible reconstruction.
    filenames = {row["filename"] for row in per_table if row.get("cell_bbox_preprocessing", {}).get("reconstructed_empty_bboxes", 0) > 0}
    rows_map = load_requested_rows(filenames)
    for item in per_table:
        stats = item.get("cell_bbox_preprocessing", {})
        reconstructed = stats.get("reconstructed_empty_bboxes", 0)
        if reconstructed <= 0:
            continue
        row = rows_map.get(item["filename"])
        if row is None:
            continue
        num_rows, num_cols = table_shape_from_gt_html(deepcopy(row))
        product = num_rows * num_cols
        if product > 90:
            continue
        candidates.append(
            {
                "sample": item["sample"],
                "filename": item["filename"],
                "reconstructed_empty_bboxes": reconstructed,
                "num_rows": num_rows,
                "num_cols": num_cols,
                "grid_cells": product,
            }
        )

    # Spread over different reconstruction sizes while keeping examples readable.
    candidates = sorted(candidates, key=lambda x: (x["reconstructed_empty_bboxes"], x["grid_cells"]))
    if len(candidates) < 5:
        return candidates[:5]
    idxs = [0, len(candidates) // 4, len(candidates) // 2, (3 * len(candidates)) // 4, len(candidates) - 1]
    chosen = []
    seen = set()
    for idx in idxs:
        item = candidates[idx]
        if item["sample"] not in seen:
            chosen.append(item)
            seen.add(item["sample"])
    return chosen[:5]


def crop_from_boxes(boxes: list[tuple[float, float, float, float]], image_size: tuple[int, int]):
    if not boxes:
        return (0, 0, image_size[0], image_size[1])
    x0 = max(0, math.floor(min(b[0] for b in boxes) - 24))
    y0 = max(0, math.floor(min(b[1] for b in boxes) - 24))
    x1 = min(image_size[0], math.ceil(max(b[2] for b in boxes) + 24))
    y1 = min(image_size[1], math.ceil(max(b[3] for b in boxes) + 24))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, image_size[0], image_size[1])
    return (x0, y0, x1, y1)


def draw_overlay(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    title: str,
    boxes_blue: list[tuple[float, float, float, float]],
    boxes_orange: list[tuple[float, float, float, float]] | None = None,
) -> Image.Image:
    boxes_orange = boxes_orange or []
    cropped = image.crop(crop_box).convert("RGB")
    scale = min(2.5, max(1.0, 900 / max(cropped.width, cropped.height)))
    if scale > 1.01:
        cropped = cropped.resize((int(cropped.width * scale), int(cropped.height * scale)))
    draw = ImageDraw.Draw(cropped)
    ox, oy = crop_box[0], crop_box[1]
    for bbox in boxes_blue:
        x0, y0, x1, y1 = [(v - (ox if i % 2 == 0 else oy)) * scale for i, v in enumerate(bbox)]
        draw.rectangle((x0, y0, x1, y1), outline="#1f77b4", width=2)
    for bbox in boxes_orange:
        x0, y0, x1, y1 = [(v - (ox if i % 2 == 0 else oy)) * scale for i, v in enumerate(bbox)]
        draw.rectangle((x0, y0, x1, y1), outline="#ff8c00", width=3)

    panel = Image.new("RGB", (cropped.width, cropped.height + 52), "white")
    panel.paste(cropped, (0, 52))
    pdraw = ImageDraw.Draw(panel)
    pdraw.text((8, 8), title, font=FONT_SUB, fill="black")
    pdraw.text((8, 32), "Blue = original GT cell bbox; orange = reconstructed empty-cell bbox", font=FONT_SMALL, fill="gray")
    return panel


def combine_side_by_side(left: Image.Image, right: Image.Image, title: str) -> Image.Image:
    width = left.width + right.width + 24
    height = max(left.height, right.height) + 54
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, font=FONT_TITLE, fill="black")
    canvas.paste(left, (0, 54))
    canvas.paste(right, (left.width + 24, 54))
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = choose_samples()
    rows_map = load_requested_rows({s["filename"] for s in samples})
    summary = []
    for sample in samples:
        row = deepcopy(rows_map[sample["filename"]])
        image_path = PUBTABNET_IMAGES / sample["filename"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        prepared, stats = EXP.reconstruct_missing_cell_bboxes(deepcopy(row), width, height)

        original_boxes = [
            bbox
            for bbox in (bbox_from_cell(cell) for cell in row["html"]["cells"])
            if bbox is not None
        ]
        reconstructed_boxes = [
            bbox
            for bbox in (bbox_from_cell(cell) for cell in prepared["html"]["cells"])
            if bbox is not None
        ]
        orange_boxes = [
            bbox
            for bbox in (bbox_from_cell(cell) for cell in prepared["html"]["cells"] if cell.get("bbox_reconstructed"))
            if bbox is not None
        ]
        crop_box = crop_from_boxes(reconstructed_boxes or original_boxes, image.size)

        original_panel = draw_overlay(
            image,
            crop_box,
            f"{sample['sample']} | PTN original GT bboxes ({len(original_boxes)})",
            original_boxes,
        )
        reconstructed_panel = draw_overlay(
            image,
            crop_box,
            f"{sample['sample']} | PTN reconstructed GT bboxes ({len(reconstructed_boxes)})",
            [bbox for bbox in reconstructed_boxes if bbox not in orange_boxes],
            orange_boxes,
        )
        original_path = OUT_DIR / "original_gt" / f"{sample['sample']}_original_gt_bboxes.png"
        reconstructed_path = OUT_DIR / "reconstructed_gt" / f"{sample['sample']}_reconstructed_gt_bboxes.png"
        combined_path = OUT_DIR / "side_by_side" / f"{sample['sample']}_original_vs_reconstructed_gt_bboxes.png"
        original_path.parent.mkdir(parents=True, exist_ok=True)
        reconstructed_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        original_panel.save(original_path)
        reconstructed_panel.save(reconstructed_path)
        combine_side_by_side(
            original_panel,
            reconstructed_panel,
            f"{sample['sample']} | {sample['num_rows']}x{sample['num_cols']} | reconstructed empty bboxes={stats['reconstructed_empty_bboxes']}",
        ).save(combined_path)

        summary.append(
            {
                **sample,
                "original_gt_boxes": len(original_boxes),
                "reconstructed_gt_boxes": len(reconstructed_boxes),
                "reconstructed_only_boxes": len(orange_boxes),
                "original_path": str(original_path),
                "reconstructed_path": str(reconstructed_path),
                "combined_path": str(combined_path),
            }
        )

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {len(summary)} samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
