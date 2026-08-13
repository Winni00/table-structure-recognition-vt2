"""Build a tiny TFLOP aux bundle for FinTabNet using PDF text extraction.

This is an adaptation to approximate the OCR-free PDF parsing described in TFLOP.
It renders pages using the PDF mediabox and extracts word boxes via PDFium text.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pypdfium2 as pdfium
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextContainer, LTTextLine
import pdfplumber


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
KAGGLE_ROOT = BASE_DIR / "data" / "fintabnet_kaggle"
DEFAULT_TABLE_JSONL = KAGGLE_ROOT / "FinTabNet_1.0.0_cell_val.jsonl"
DEFAULT_PDF_DIR = KAGGLE_ROOT / "pdfs"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "tflop_fintabnet_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-jsonl", type=Path, default=DEFAULT_TABLE_JSONL)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--render-scale", type=float, default=2.0)
    parser.add_argument(
        "--use-pdfminer-lines",
        action="store_true",
        help="Use pdfminer to extract line-level text boxes (PDF parsing) instead of PDFium char boxes.",
    )
    parser.add_argument(
        "--use-pdfminer-blocks",
        action="store_true",
        help="Use pdfminer to extract block-level text boxes (PDF parsing) instead of PDFium char boxes.",
    )
    parser.add_argument(
        "--use-region-clusters",
        action="store_true",
        help="Cluster line boxes into text regions (columns + vertical grouping).",
    )
    parser.add_argument(
        "--use-pdfplumber-layout",
        action="store_true",
        help="Use pdfplumber layout extraction to build OCR-like text regions.",
    )
    parser.add_argument(
        "--col-x-tol",
        type=float,
        default=40.0,
        help="Column grouping tolerance for line-box x0 (image pixels).",
    )
    parser.add_argument(
        "--line-gap-tol",
        type=float,
        default=8.0,
        help="Max vertical gap between lines to merge into a region (image pixels).",
    )
    parser.add_argument(
        "--pdfminer-laparams",
        default="",
        help="Optional pdfminer LAParams overrides (unused placeholder for now).",
    )
    parser.add_argument(
        "--group-lines",
        action="store_true",
        help="Group word boxes into line-level regions (closer to TFLOP text-region inputs).",
    )
    parser.add_argument(
        "--line-y-tol",
        type=float,
        default=6.0,
        help="Y-distance tolerance (in image pixels) when grouping words into a line.",
    )
    return parser.parse_args()


def ensure_html_table_document(text: str) -> str:
    cleaned = text.strip()
    if "<html" in cleaned.lower():
        return cleaned
    if "<table" in cleaned.lower():
        return f"<html><body>{cleaned}</body></html>"
    return f"<html><body><table>{cleaned}</table></body></html>"


def rich_text(tokens: list[str]) -> str:
    return "".join(tokens or [])


def html_from_fintabnet(row: dict[str, Any]) -> str:
    cells = row["html"]["cells"]
    cell_iter = iter(cells)
    parts: list[str] = []
    for token in row["html"]["structure"]["tokens"]:
        if token == "</td>":
            cell = next(cell_iter, {})
            parts.append(rich_text(cell.get("tokens", [])))
            parts.append(token)
        else:
            parts.append(token)
    return ensure_html_table_document("".join(parts))


def render_with_mediabox(
    pdf_path: Path, scale: float
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    doc = pdfium.PdfDocument(str(pdf_path))
    page = doc[0]
    mediabox = page.get_mediabox()
    cropbox = page.get_cropbox()
    page.set_cropbox(*mediabox)
    image = page.render(scale=scale).to_pil().convert("RGB")
    page.set_cropbox(*cropbox)
    arr = np.array(image)
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise RuntimeError(f"Empty render for {pdf_path}")
    return arr, mediabox


def table_bbox_to_image_coords(
    bbox: list[float],
    mediabox: tuple[float, float, float, float],
    scale: float,
) -> tuple[int, int, int, int]:
    """Convert a PDF-space table bbox into image-space crop coords (left, top, right, bottom)."""
    x0, y0, x1, y1 = bbox
    mb_x0, mb_y0, mb_x1, mb_y1 = mediabox
    left = (x0 - mb_x0) * scale
    right = (x1 - mb_x0) * scale
    top = (mb_y1 - y1) * scale
    bottom = (mb_y1 - y0) * scale
    return (
        int(max(0.0, left)),
        int(max(0.0, top)),
        int(max(0.0, right)),
        int(max(0.0, bottom)),
    )


def iter_word_boxes(textpage: pdfium.PdfTextPage) -> Iterable[tuple[str, tuple[float, float, float, float]]]:
    n = textpage.count_chars()
    current = []
    boxes = []
    for i in range(n):
        ch = textpage.get_text_range(i, 1)
        if not ch or ch.isspace():
            if current:
                word = "".join(current)
                xs = [b[0] for b in boxes] + [b[2] for b in boxes]
                ys = [b[1] for b in boxes] + [b[3] for b in boxes]
                yield word, (min(xs), min(ys), max(xs), max(ys))
                current, boxes = [], []
            continue
        current.append(ch)
        boxes.append(textpage.get_charbox(i))
    if current:
        word = "".join(current)
        xs = [b[0] for b in boxes] + [b[2] for b in boxes]
        ys = [b[1] for b in boxes] + [b[3] for b in boxes]
        yield word, (min(xs), min(ys), max(xs), max(ys))


def iter_pdfminer_lines(pdf_path: Path) -> Iterable[tuple[str, tuple[float, float, float, float]]]:
    """Yield (text, bbox) for each pdfminer text line in PDF coordinates."""
    for page_layout in extract_pages(str(pdf_path)):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for line in element:
                    if not isinstance(line, LTTextLine):
                        continue
                    text = line.get_text().strip()
                    if not text:
                        continue
                    x0, y0, x1, y1 = line.bbox
                    yield text, (x0, y0, x1, y1)


def iter_pdfminer_blocks(pdf_path: Path) -> Iterable[tuple[str, tuple[float, float, float, float]]]:
    """Yield (text, bbox) for each pdfminer text block in PDF coordinates."""
    for page_layout in extract_pages(str(pdf_path)):
        for element in page_layout:
            if not isinstance(element, LTTextContainer):
                continue
            if isinstance(element, LTTextBox):
                text = element.get_text().strip()
                if not text:
                    continue
                x0, y0, x1, y1 = element.bbox
                yield text, (x0, y0, x1, y1)


def iter_pdfplumber_regions(
    pdf_path: Path,
    table_bbox: list[float],
    mediabox: tuple[float, float, float, float],
    scale: float,
) -> list[dict[str, Any]]:
    """Extract OCR-like text regions using pdfplumber and cluster by column + gaps."""
    mb_x0, mb_y0, mb_x1, mb_y1 = mediabox
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[0]
        words = page.extract_words(
            x_tolerance=2,
            y_tolerance=2,
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=[],
        )

    # Filter to table bbox in PDF coordinates (bottom-left)
    tb_x0, tb_y0, tb_x1, tb_y1 = table_bbox
    filtered = []
    for w in words:
        x0, x1 = w["x0"], w["x1"]
        y0, y1 = w["bottom"], w["top"]
        if x1 < tb_x0 or x0 > tb_x1 or y1 < tb_y0 or y0 > tb_y1:
            continue
        filtered.append(
            {
                "text": w["text"],
                "bbox": [x0, y0, x1, y1],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "cy": (y0 + y1) / 2.0,
            }
        )

    if not filtered:
        return []

    # Column clustering by x0
    filtered.sort(key=lambda w: w["x0"])
    columns: list[dict[str, Any]] = []
    col_tol = 20.0
    for w in filtered:
        placed = False
        for col in columns:
            if abs(w["x0"] - col["x_mean"]) <= col_tol:
                col["words"].append(w)
                col["x_mean"] = sum(x["x0"] for x in col["words"]) / len(col["words"])
                placed = True
                break
        if not placed:
            columns.append({"x_mean": w["x0"], "words": [w]})

    regions: list[dict[str, Any]] = []
    for col in columns:
        col_words = sorted(col["words"], key=lambda w: w["y0"])
        current: list[dict[str, Any]] = []
        last_y1 = None
        gap_tol = 6.0
        for w in col_words:
            if last_y1 is None or (w["y0"] - last_y1) <= gap_tol:
                current.append(w)
            else:
                regions.append(_merge_word_group(current))
                current = [w]
            last_y1 = w["y1"]
        if current:
            regions.append(_merge_word_group(current))

    # Convert to image-space coords and crop offsets later in pdf_words_to_image_coords
    # Return as list of dicts with pdf-space bbox + text
    return [{"bbox": r["bbox"], "text": r["text"]} for r in regions]


def _merge_word_group(words: list[dict[str, Any]]) -> dict[str, Any]:
    xs = [w["x0"] for w in words] + [w["x1"] for w in words]
    ys = [w["y0"] for w in words] + [w["y1"] for w in words]
    text = " ".join(w["text"] for w in words if w["text"]).strip()
    return {"bbox": [min(xs), min(ys), max(xs), max(ys)], "text": text}


def pdf_words_to_image_coords(
    words: Iterable[tuple[str, tuple[float, float, float, float]]],
    mediabox: tuple[float, float, float, float],
    scale: float,
    image_width: int,
    image_height: int,
    table_bbox: list[float] | None = None,
    crop_offset: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = mediabox
    items: list[dict[str, Any]] = []
    tb_x0 = tb_y0 = tb_x1 = tb_y1 = None
    if table_bbox is not None:
        tb_x0, tb_y0, tb_x1, tb_y1 = table_bbox
    offset_x, offset_y = crop_offset or (0, 0)

    for text, (l, b, r, t) in words:
        if not text.strip():
            continue
        if table_bbox is not None:
            # Only keep words inside the table bbox in PDF coordinates.
            if r < tb_x0 or l > tb_x1 or t < tb_y0 or b > tb_y1:
                continue
        # map PDF bottom-left coords to image top-left coords
        left = (l - x0) * scale - offset_x
        right = (r - x0) * scale - offset_x
        top = (y1 - t) * scale - offset_y
        bottom = (y1 - b) * scale - offset_y
        # Clamp to image bounds to avoid ROIAlign index errors.
        left = max(0.0, min(left, image_width))
        right = max(0.0, min(right, image_width))
        top = max(0.0, min(top, image_height))
        bottom = max(0.0, min(bottom, image_height))
        if right - left < 1.0 or bottom - top < 1.0:
            continue
        items.append({"bbox": [left, top, right, bottom], "text": text})
    return items


def group_words_into_lines(
    rec_items: list[dict[str, Any]], y_tol: float
) -> list[dict[str, Any]]:
    """Group word boxes into line-level boxes to approximate text regions."""
    if not rec_items:
        return rec_items

    words = [
        {
            "bbox": item["bbox"],
            "text": item.get("text", ""),
            "cy": (item["bbox"][1] + item["bbox"][3]) / 2.0,
        }
        for item in rec_items
    ]
    words.sort(key=lambda w: (w["cy"], w["bbox"][0]))

    lines: list[dict[str, Any]] = []
    for word in words:
        placed = False
        for line in lines:
            if abs(word["cy"] - line["cy"]) <= y_tol:
                line["words"].append(word)
                line["cy"] = sum(w["cy"] for w in line["words"]) / len(line["words"])
                placed = True
                break
        if not placed:
            lines.append({"cy": word["cy"], "words": [word]})

    merged: list[dict[str, Any]] = []
    for line in lines:
        line_words = sorted(line["words"], key=lambda w: w["bbox"][0])
        xs = [w["bbox"][0] for w in line_words] + [w["bbox"][2] for w in line_words]
        ys = [w["bbox"][1] for w in line_words] + [w["bbox"][3] for w in line_words]
        text = " ".join(w["text"] for w in line_words if w["text"]).strip()
        if not text:
            continue
        merged.append(
            {
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "text": text,
            }
        )
    return merged


def cluster_lines_into_regions(
    rec_items: list[dict[str, Any]],
    col_x_tol: float,
    line_gap_tol: float,
) -> list[dict[str, Any]]:
    """Cluster line boxes into column-aligned text regions."""
    if not rec_items:
        return rec_items

    lines = [
        {
            "bbox": item["bbox"],
            "text": item.get("text", ""),
            "x0": item["bbox"][0],
            "y0": item["bbox"][1],
            "x1": item["bbox"][2],
            "y1": item["bbox"][3],
        }
        for item in rec_items
    ]
    lines.sort(key=lambda l: l["x0"])

    columns: list[dict[str, Any]] = []
    for line in lines:
        assigned = False
        for col in columns:
            if abs(line["x0"] - col["x_mean"]) <= col_x_tol:
                col["lines"].append(line)
                col["x_mean"] = sum(l["x0"] for l in col["lines"]) / len(col["lines"])
                assigned = True
                break
        if not assigned:
            columns.append({"x_mean": line["x0"], "lines": [line]})

    regions: list[dict[str, Any]] = []
    for col in columns:
        col_lines = sorted(col["lines"], key=lambda l: l["y0"])
        current: list[dict[str, Any]] = []
        last_y1 = None
        for line in col_lines:
            if last_y1 is None or (line["y0"] - last_y1) <= line_gap_tol:
                current.append(line)
            else:
                regions.append(_merge_lines(current))
                current = [line]
            last_y1 = line["y1"]
        if current:
            regions.append(_merge_lines(current))

    return regions


def _merge_lines(lines: list[dict[str, Any]]) -> dict[str, Any]:
    xs = [l["x0"] for l in lines] + [l["x1"] for l in lines]
    ys = [l["y0"] for l in lines] + [l["y1"] for l in lines]
    text = " ".join(l["text"] for l in lines if l["text"]).strip()
    return {"bbox": [min(xs), min(ys), max(xs), max(ys)], "text": text}


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
    return safe.strip("_") or "page"


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.table_jsonl.open()]
    subset = rows[args.offset : args.offset + args.limit]

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    aux_json: dict[str, dict[str, Any]] = {}
    aux_rec: dict[str, list[dict[str, Any]]] = {}

    for idx, row in enumerate(subset):
        pdf_path = args.pdf_dir / row["filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF: {pdf_path}")

        image, mediabox = render_with_mediabox(pdf_path, args.render_scale)
        table_bbox = row.get("bbox")
        if not table_bbox:
            raise ValueError(f"Missing table bbox in row: {row.get('table_id')}")
        crop_left, crop_top, crop_right, crop_bottom = table_bbox_to_image_coords(
            table_bbox, mediabox, args.render_scale
        )
        # Crop table region from the rendered page
        image_crop = image[crop_top:crop_bottom, crop_left:crop_right]
        if image_crop.size == 0 or image_crop.shape[0] == 0 or image_crop.shape[1] == 0:
            raise RuntimeError(f"Empty table crop for {row.get('table_id')} from {pdf_path}")

        if args.use_pdfplumber_layout:
            words = [
                (item["text"], tuple(item["bbox"]))
                for item in iter_pdfplumber_regions(pdf_path, table_bbox, mediabox, args.render_scale)
            ]
        elif args.use_pdfminer_lines:
            words = list(iter_pdfminer_lines(pdf_path))
        elif args.use_pdfminer_blocks:
            words = list(iter_pdfminer_blocks(pdf_path))
        else:
            textpage = pdfium.PdfDocument(str(pdf_path))[0].get_textpage()
            words = list(iter_word_boxes(textpage))

        if not words:
            # Skip pages with no extracted text regions to avoid TFLOP crashing.
            continue
        rec_items = pdf_words_to_image_coords(
            words,
            mediabox,
            args.render_scale,
            image_width=image_crop.shape[1],
            image_height=image_crop.shape[0],
            table_bbox=table_bbox,
            crop_offset=(crop_left, crop_top),
        )
        if args.group_lines:
            rec_items = group_words_into_lines(rec_items, y_tol=args.line_y_tol)
        if args.use_region_clusters:
            rec_items = cluster_lines_into_regions(
                rec_items,
                col_x_tol=args.col_x_tol,
                line_gap_tol=args.line_gap_tol,
            )

        out_name = f"fintabnet_{idx:04d}_{sanitize_filename(row['table_id'])}.png"
        out_path = images_dir / out_name
        from PIL import Image

        Image.fromarray(image_crop).save(out_path)

        aux_json[out_name] = {
            "html": html_from_fintabnet(row),
            "type": "simple",
        }
        aux_rec[out_name] = rec_items

    (args.output_dir / "aux.json").write_text(json.dumps(aux_json, indent=2), encoding="utf-8")
    with (args.output_dir / "aux_rec.pkl").open("wb") as f:
        pickle.dump(aux_rec, f)

    summary = {
        "samples": len(subset),
        "output_dir": str(args.output_dir),
        "images_dir": str(images_dir),
        "aux_json": str(args.output_dir / "aux.json"),
        "aux_rec": str(args.output_dir / "aux_rec.pkl"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
