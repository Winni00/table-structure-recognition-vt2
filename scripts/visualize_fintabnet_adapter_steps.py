#!/usr/bin/env python3
"""Visualize the FinTabNet -> TFLOP aux adapter steps."""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path("/cluster/home/trinhwin/vt2/docling")
sys.path.insert(0, str(ROOT))

from build_tflop_fintabnet_annotation_aux import (
    cell_bbox_to_crop_coords,
    rec_from_fintabnet_row,
    rich_text,
)
from build_tflop_fintabnet_smoke_aux import (
    html_from_fintabnet,
    render_with_mediabox,
    table_bbox_to_image_coords,
)


FTN_JSONL = ROOT / "data" / "fintabnet_kaggle" / "FinTabNet_1.0.0_cell_val_excluding_28.jsonl"
PDF_DIR = ROOT / "data" / "fintabnet_kaggle" / "pdfs"
RUN_DIR = ROOT / "results" / "tflop_fintabnet_full_annotation_excl_problem_pages"
OUT_DIR = ROOT / "results" / "fintabnet_adapter_steps_visualized"
SCALE = 2.0


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
    return safe.strip("_") or "page"


def text_label(text: str, limit: int = 30) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = " ".join(text.split())
    text = text.encode("ascii", "replace").decode("ascii")
    return text[:limit] + ("..." if len(text) > limit else "")


def draw_origin(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, max_x: int, max_y: int) -> None:
    draw.ellipse([x, y, x + 10, y + 10], fill=(220, 40, 40))
    draw.line([x + 5, y + 5, min(max_x, x + 110), y + 5], fill=(220, 40, 40), width=3)
    draw.line([x + 5, y + 5, x + 5, min(max_y, y + 90)], fill=(220, 40, 40), width=3)
    draw.text((x + 13, y + 8), label, fill=(220, 40, 40))
    draw.text((min(max_x - 20, x + 115), y), "x", fill=(220, 40, 40))
    draw.text((x + 12, min(max_y - 16, y + 95)), "y", fill=(220, 40, 40))


def draw_boxes_on_image(
    img: Image.Image,
    boxes: list[dict[str, Any]],
    title: str,
    subtitle: str,
    boundary: list[float] | None,
    out_path: Path,
) -> None:
    w, h = img.size
    max_w = 1300
    view_scale = min(1.0, max_w / max(w, 1))
    view = img.resize((int(w * view_scale), int(h * view_scale)))
    top = 116
    canvas = Image.new("RGB", (view.width, view.height + top + 28), "white")
    canvas.paste(view, (0, top))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), title, fill=(0, 0, 0))
    d.text((10, 30), subtitle, fill=(80, 80, 80))
    d.text((10, 52), "Visual coords: origin at top-left, x -> right, y -> down", fill=(80, 80, 80))
    d.text((10, 74), f"Image/crop size: {w} x {h}", fill=(80, 80, 80))
    d.text((10, 96), f"Boxes: {len(boxes)}", fill=(20, 130, 70))
    draw_origin(d, 0, top, "(0,0)", view.width - 1, top + view.height - 1)

    if boundary:
        x0, y0, x1, y1 = [v * view_scale for v in boundary]
        d.rectangle([x0, top + y0, x1, top + y1], outline=(35, 95, 220), width=4)
        d.text((x0 + 4, top + y0 + 4), "table bbox", fill=(35, 95, 220))

    for item in boxes:
        bbox = item["bbox"]
        x0, y0, x1, y1 = [float(v) * view_scale for v in bbox]
        d.rectangle([x0, top + y0, x1, top + y1], outline=(20, 150, 70), width=2)
        label = text_label(item.get("text", ""))
        if label and x1 - x0 > 20:
            d.rectangle([x0, max(top, top + y0 - 12), min(view.width, x0 + 6 * len(label) + 8), max(top, top + y0 - 12) + 12], fill="white")
            d.text((x0 + 2, max(top, top + y0 - 12)), label, fill=(20, 120, 60))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def render_one(row: dict[str, Any], idx: int) -> dict[str, Any]:
    pdf_path = PDF_DIR / row["filename"]
    table_id = sanitize_filename(row["table_id"])
    name = f"{idx:02d}_{table_id}"
    image_arr, mediabox = render_with_mediabox(pdf_path, SCALE)
    page_img = Image.fromarray(image_arr)
    table_bbox_pdf = row["bbox"]
    crop_left, crop_top, crop_right, crop_bottom = table_bbox_to_image_coords(table_bbox_pdf, mediabox, SCALE)
    crop = page_img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Full-page cell boxes in rendered image coordinates.
    page_cell_boxes = []
    crop_cell_boxes = []
    for cell in row["html"]["cells"]:
        bbox = cell.get("bbox")
        text = rich_text(cell.get("tokens", []))
        if not bbox or len(bbox) != 4 or not text:
            continue
        full_bbox = table_bbox_to_image_coords(bbox, mediabox, SCALE)
        page_cell_boxes.append({"bbox": full_bbox, "text": text})
        crop_bbox = cell_bbox_to_crop_coords(bbox, mediabox, SCALE, (crop_left, crop_top), crop.size)
        if crop_bbox:
            crop_cell_boxes.append({"bbox": crop_bbox, "text": text})

    rec_items = rec_from_fintabnet_row(row, mediabox, SCALE, (crop_left, crop_top), crop.size)
    out_dir = OUT_DIR / name
    page_path = out_dir / "01_rendered_pdf_page_table_bbox_and_original_cell_boxes.png"
    crop_path = out_dir / "02_table_crop_transformed_text_region_boxes.png"
    aux_json_path = out_dir / "03_aux_json_entry.json"
    aux_rec_path = out_dir / "04_aux_rec_preview.json"
    notes_path = out_dir / "README.md"

    draw_boxes_on_image(
        page_img,
        page_cell_boxes,
        f"Step 1-2: Render FTN PDF page + table bbox ({row['filename']})",
        "Blue=table bbox in rendered page coords. Green=FTN cell boxes transformed from PDF coords to rendered page coords.",
        [crop_left, crop_top, crop_right, crop_bottom],
        page_path,
    )
    draw_boxes_on_image(
        crop,
        [{"bbox": item["bbox"], "text": item["text"]} for item in rec_items],
        f"Step 3-4: Save table crop + transform cell boxes ({row['table_id']})",
        "Green=TFLOP input text-region boxes in crop coords. This becomes aux_rec.pkl.",
        [0, 0, crop.size[0], crop.size[1]],
        crop_path,
    )

    aux_json = {
        f"fintabnet_{idx:05d}_{table_id}.png": {
            "html": html_from_fintabnet(row),
            "type": "simple",
        }
    }
    aux_json_path.write_text(json.dumps(aux_json, indent=2, ensure_ascii=False), encoding="utf-8")
    aux_rec_path.write_text(json.dumps(rec_items[:30], indent=2, ensure_ascii=False), encoding="utf-8")

    notes = f"""# FinTabNet Adapter Step Visualization

Sample: `{row['filename']}` / table_id `{row['table_id']}`

Pipeline shown here:

1. FinTabNet Kaggle PDF + FTN cell annotations
2. PDF rendered with mediabox at scale `{SCALE}`
3. Table bbox transformed from PDF coordinates to rendered image coordinates
4. Table crop saved
5. Cell bbox/text transformed into crop coordinates
6. `aux.json` and `aux_rec.pkl` style entries generated

Coordinate systems:

- Original FTN annotations are PDF coordinates.
- Rendered page/crop visualizations use image coordinates: origin `(0,0)` top-left, x right, y down.
- Crop coordinates subtract table crop offset from rendered page coordinates.

Files:

- `01_rendered_pdf_page_table_bbox_and_original_cell_boxes.png`
- `02_table_crop_transformed_text_region_boxes.png`
- `03_aux_json_entry.json`
- `04_aux_rec_preview.json`
"""
    notes_path.write_text(notes, encoding="utf-8")
    return {
        "sample": row["filename"],
        "table_id": row["table_id"],
        "pdf": str(pdf_path),
        "mediabox": list(mediabox),
        "table_bbox_pdf": table_bbox_pdf,
        "table_bbox_rendered_page": [crop_left, crop_top, crop_right, crop_bottom],
        "crop_size": list(crop.size),
        "cell_boxes_with_text": len(crop_cell_boxes),
        "aux_rec_items": len(rec_items),
        "page_png": str(page_path),
        "crop_png": str(crop_path),
        "aux_json": str(aux_json_path),
        "aux_rec_preview": str(aux_rec_path),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with FTN_JSONL.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if (PDF_DIR / row["filename"]).exists() and row.get("bbox"):
                rows.append(row)
            if len(rows) >= 5:
                break
    manifest = {"output_dir": str(OUT_DIR), "samples": []}
    for i, row in enumerate(rows):
        manifest["samples"].append(render_one(row, i))
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# FinTabNet Adapter Steps Visualized", ""]
    for sample in manifest["samples"]:
        lines.append(f"## {sample['table_id']} ({sample['sample']})")
        lines.append(f"- Rendered page/table bbox: `{sample['page_png']}`")
        lines.append(f"- Crop/text-region boxes: `{sample['crop_png']}`")
        lines.append(f"- aux.json entry: `{sample['aux_json']}`")
        lines.append(f"- aux_rec preview: `{sample['aux_rec_preview']}`")
        lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR), "readme": str(OUT_DIR / "README.md")}, indent=2))


if __name__ == "__main__":
    main()
