"""Build TFLOP aux inputs from FinTabNet Kaggle cell annotations.

This renders FinTabNet table crops from PDFs and feeds annotated cell boxes/text
as TFLOP recognition inputs. It is an oracle-input counterpart to the
pdfplumber adapter and is useful for checking TFLOP on FinTabNet without OCR or
PDF text extraction noise.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

from PIL import Image

from build_tflop_fintabnet_smoke_aux import (
    html_from_fintabnet,
    render_with_mediabox,
    table_bbox_to_image_coords,
)


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
KAGGLE_ROOT = BASE_DIR / "data" / "fintabnet_kaggle"
DEFAULT_JSONL = KAGGLE_ROOT / "FinTabNet_1.0.0_cell_val_excluding_28.jsonl"
DEFAULT_PDF_DIR = KAGGLE_ROOT / "pdfs"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "tflop_fintabnet_full_annotation_excl_problem_pages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--render-scale", type=float, default=2.0)
    return parser.parse_args()


def rich_text(tokens: list[str]) -> str:
    return "".join(tokens or [])


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
    return safe.strip("_") or "page"


def cell_bbox_to_crop_coords(
    bbox: list[float],
    mediabox: tuple[float, float, float, float],
    scale: float,
    crop_offset: tuple[int, int],
    crop_size: tuple[int, int],
) -> list[float] | None:
    left, top, right, bottom = table_bbox_to_image_coords(bbox, mediabox, scale)
    crop_left, crop_top = crop_offset
    width, height = crop_size
    left -= crop_left
    right -= crop_left
    top -= crop_top
    bottom -= crop_top
    left = max(0, min(float(left), float(width)))
    right = max(0, min(float(right), float(width)))
    top = max(0, min(float(top), float(height)))
    bottom = max(0, min(float(bottom), float(height)))
    if right - left < 1 or bottom - top < 1:
        return None
    return [left, top, right, bottom]


def rec_from_fintabnet_row(
    row: dict[str, Any],
    mediabox: tuple[float, float, float, float],
    scale: float,
    crop_offset: tuple[int, int],
    crop_size: tuple[int, int],
) -> list[dict[str, Any]]:
    rec_items: list[dict[str, Any]] = []
    for cell in row["html"]["cells"]:
        bbox = cell.get("bbox")
        text = rich_text(cell.get("tokens", []))
        if not bbox or len(bbox) != 4 or not text:
            continue
        crop_bbox = cell_bbox_to_crop_coords(bbox, mediabox, scale, crop_offset, crop_size)
        if crop_bbox is None:
            continue
        rec_items.append({"bbox": crop_bbox, "text": text, "score": 1.0})
    return rec_items


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.table_jsonl.open(encoding="utf-8")]
    rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]

    aux_json: dict[str, dict[str, Any]] = {}
    aux_rec: dict[str, list[dict[str, Any]]] = {}
    skipped_missing_pdf = 0
    skipped_empty_rec = 0

    for idx, row in enumerate(rows):
        pdf_path = args.pdf_dir / row["filename"]
        if not pdf_path.exists():
            skipped_missing_pdf += 1
            continue
        table_bbox = row.get("bbox")
        if not table_bbox:
            skipped_empty_rec += 1
            continue

        image, mediabox = render_with_mediabox(pdf_path, args.render_scale)
        crop_left, crop_top, crop_right, crop_bottom = table_bbox_to_image_coords(
            table_bbox, mediabox, args.render_scale
        )
        image_crop = image[crop_top:crop_bottom, crop_left:crop_right]
        if image_crop.size == 0 or image_crop.shape[0] == 0 or image_crop.shape[1] == 0:
            skipped_empty_rec += 1
            continue

        rec_items = rec_from_fintabnet_row(
            row,
            mediabox,
            args.render_scale,
            (crop_left, crop_top),
            (image_crop.shape[1], image_crop.shape[0]),
        )
        if not rec_items:
            skipped_empty_rec += 1
            continue

        out_name = f"fintabnet_{idx:05d}_{sanitize_filename(row['table_id'])}.png"
        Image.fromarray(image_crop).save(images_dir / out_name)
        aux_json[out_name] = {"html": html_from_fintabnet(row), "type": "simple"}
        aux_rec[out_name] = rec_items

    aux_json_path = args.output_dir / "aux.json"
    aux_rec_path = args.output_dir / "aux_rec.pkl"
    aux_json_path.write_text(json.dumps(aux_json, ensure_ascii=False), encoding="utf-8")
    with aux_rec_path.open("wb") as f:
        pickle.dump(aux_rec, f)

    summary = {
        "samples_requested": len(rows),
        "samples": len(aux_json),
        "table_jsonl": str(args.table_jsonl),
        "pdf_dir": str(args.pdf_dir),
        "images_dir": str(images_dir),
        "aux_json": str(aux_json_path),
        "aux_rec": str(aux_rec_path),
        "skipped_missing_pdf": skipped_missing_pdf,
        "skipped_empty_rec": skipped_empty_rec,
        "adapter": "FinTabNet annotated cell bbox/text oracle input",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
