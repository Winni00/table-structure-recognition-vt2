#!/usr/bin/env python3
"""Render the excluded FinTabNet problem pages with annotation boxes."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path("/cluster/home/trinhwin/vt2/docling")
sys.path.insert(0, str(ROOT))

from build_tflop_fintabnet_annotation_aux import rich_text
from build_tflop_fintabnet_smoke_aux import render_with_mediabox, table_bbox_to_image_coords


FULL_JSONL = ROOT / "data" / "fintabnet_kaggle" / "FinTabNet_1.0.0_cell_val.jsonl"
PROBLEM_LIST = ROOT / "data" / "fintabnet_problem_pages.txt"
PDF_DIR = ROOT / "data" / "fintabnet_kaggle" / "pdfs"
OUT_DIR = ROOT / "results" / "fintabnet_problem_pages_visualized"
SCALE = 2.0


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def label(text: str, limit: int = 32) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = " ".join(text.split())
    text = text.encode("ascii", "replace").decode("ascii")
    return text[:limit] + ("..." if len(text) > limit else "")


def draw_origin(draw: ImageDraw.ImageDraw, x: int, y: int, max_x: int, max_y: int) -> None:
    draw.ellipse([x, y, x + 10, y + 10], fill=(220, 40, 40))
    draw.line([x + 5, y + 5, min(max_x, x + 105), y + 5], fill=(220, 40, 40), width=3)
    draw.line([x + 5, y + 5, x + 5, min(max_y, y + 80)], fill=(220, 40, 40), width=3)
    draw.text((x + 12, y + 8), "(0,0)", fill=(220, 40, 40))
    draw.text((min(max_x - 15, x + 110), y), "x", fill=(220, 40, 40))
    draw.text((x + 12, min(max_y - 14, y + 84)), "y", fill=(220, 40, 40))


def draw_page(
    image: Image.Image,
    title: str,
    subtitle: str,
    table_bbox: list[int],
    cell_boxes: list[dict[str, Any]],
    out_path: Path,
) -> None:
    max_w = 1450
    s = min(1.0, max_w / max(image.width, 1))
    view = image.resize((int(image.width * s), int(image.height * s)))
    header = 110
    canvas = Image.new("RGB", (view.width, view.height + header + 12), "white")
    canvas.paste(view, (0, header))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), title, fill=(0, 0, 0))
    d.text((8, 31), subtitle, fill=(80, 80, 80))
    d.text((8, 53), "Blue=table bbox, green=annotated cell boxes/text, red=visual coordinate origin", fill=(80, 80, 80))
    d.text((8, 75), f"Image size: {image.width}x{image.height}; cells with bbox/text: {len(cell_boxes)}", fill=(80, 80, 80))
    draw_origin(d, 0, header, view.width - 1, header + view.height - 1)

    x0, y0, x1, y1 = [v * s for v in table_bbox]
    d.rectangle([x0, header + y0, x1, header + y1], outline=(35, 95, 220), width=4)
    d.text((x0 + 4, header + y0 + 4), "table bbox", fill=(35, 95, 220))

    for item in cell_boxes:
        bx0, by0, bx1, by1 = [float(v) * s for v in item["bbox"]]
        d.rectangle([bx0, header + by0, bx1, header + by1], outline=(15, 150, 70), width=2)
        txt = label(item.get("text", ""))
        if txt and bx1 - bx0 > 16:
            d.text((bx0 + 2, max(header, header + by0 - 12)), txt, fill=(15, 120, 55))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def load_problem_rows() -> list[dict[str, Any]]:
    problem_files = {line.strip() for line in PROBLEM_LIST.read_text(encoding="utf-8").splitlines() if line.strip()}
    rows = []
    with FULL_JSONL.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("filename") in problem_files:
                rows.append(row)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_problem_rows()
    manifest: dict[str, Any] = {
        "problem_pdf_count": len({r["filename"] for r in rows}),
        "problem_table_samples": len(rows),
        "source_jsonl": str(FULL_JSONL),
        "problem_list": str(PROBLEM_LIST),
        "output_dir": str(OUT_DIR),
        "samples": [],
    }

    for i, row in enumerate(rows):
        pdf_path = PDF_DIR / row["filename"]
        image_arr, mediabox = render_with_mediabox(pdf_path, SCALE)
        page_img = Image.fromarray(image_arr)
        table_bbox = list(table_bbox_to_image_coords(row["bbox"], mediabox, SCALE))
        cell_boxes = []
        for cell in row["html"]["cells"]:
            bbox = cell.get("bbox")
            text = rich_text(cell.get("tokens", []))
            if not bbox or len(bbox) != 4 or not text:
                continue
            cell_boxes.append(
                {
                    "bbox": list(table_bbox_to_image_coords(bbox, mediabox, SCALE)),
                    "text": text,
                }
            )
        rel = safe_name(row["filename"].replace(".pdf", ""))
        tid = safe_name(str(row.get("table_id", i)))
        out_path = OUT_DIR / f"{i:02d}_{rel}_{tid}.png"
        draw_page(
            page_img,
            f"FinTabNet excluded problem page {i + 1}/{len(rows)}",
            f"{row['filename']} | table_id={row.get('table_id')} | page coords converted from PDF to image coords",
            table_bbox,
            cell_boxes,
            out_path,
        )
        manifest["samples"].append(
            {
                "filename": row["filename"],
                "table_id": row.get("table_id"),
                "mediabox": list(mediabox),
                "table_bbox_pdf": row.get("bbox"),
                "table_bbox_image": table_bbox,
                "cells_with_bbox_text": len(cell_boxes),
                "png": str(out_path),
            }
        )

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = ["# FinTabNet Excluded Problem Pages", ""]
    readme.append(f"- Problem PDFs: `{manifest['problem_pdf_count']}`")
    readme.append(f"- Table samples: `{manifest['problem_table_samples']}`")
    readme.append("")
    for item in manifest["samples"]:
        readme.append(f"- `{item['filename']}` / `{item['table_id']}` -> `{item['png']}`")
    (OUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR), "samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
