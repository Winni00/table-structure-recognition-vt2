"""Pilot PyMuPDF text extraction for updated paper-collection OCR boxes.

Input:
- TFLOP OCR-style run directory with `aux_rec.pkl` from PSENet+MASTER
- TargetDomain updated data with table crop coordinates in `table_coord.json`

For each PSENet text-region box in crop-image coordinates, the script maps the
box back to the PDF page using the table crop coordinates and extracts text from
that PDF rectangle with PyMuPDF. This tests whether PDF text extraction can
replace MASTER recognition for cell/text-region content.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RUN_DIR = ROOT / "results/tflop_paper_collection_updated_crops_ocr_style_public"
DEFAULT_COORD_MANIFEST = ROOT / "results/paper_collection_updated_crops_coord_check/manifest.json"
DEFAULT_OUT_DIR = ROOT / "results/paper_collection_updated_crops_pymupdf_text_pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--coord-manifest", type=Path, default=DEFAULT_COORD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--pad-points",
        type=float,
        default=1.5,
        help="Padding in PDF points around each mapped PSENet box.",
    )
    parser.add_argument(
        "--skip-header-risk",
        action="store_true",
        help="Skip TargetDomain-marked header-risk PDFs for the first pilot.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def unique_sample_id(filename: str) -> str:
    # run filename: paper_id__paper_id_table_1.png
    return Path(filename.split("__", 1)[1] if "__" in filename else filename).stem


def as_bbox(item: dict[str, Any]) -> list[float] | None:
    bbox = item.get("bbox")
    if bbox is None:
        return None
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    if len(bbox) == 4:
        return [float(v) for v in bbox]
    if len(bbox) >= 8:
        xs = [float(v) for v in bbox[0::2]]
        ys = [float(v) for v in bbox[1::2]]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def map_crop_box_to_pymupdf_rect(
    bbox: list[float],
    crop_size: tuple[int, int],
    coord: dict[str, Any],
    page_height: float,
    pad: float,
) -> fitz.Rect:
    """Map crop pixel bbox to PyMuPDF page coordinates.

    TargetDomain coordinates are PDF-style with origin at bottom-left. PyMuPDF page
    coordinates use origin at top-left. The crop image is the table rectangle
    rendered as pixels, so we scale within the crop rectangle.
    """
    crop_w, crop_h = crop_size
    x0, y0, x1, y1 = bbox
    left = float(coord["left"])
    top_bottom_origin = float(coord["top"])
    width = float(coord["width"])
    height = float(coord["height"])

    page_top = page_height - top_bottom_origin
    rx0 = left + (x0 / crop_w) * width
    rx1 = left + (x1 / crop_w) * width
    ry0 = page_top + (y0 / crop_h) * height
    ry1 = page_top + (y1 / crop_h) * height
    return fitz.Rect(rx0 - pad, ry0 - pad, rx1 + pad, ry1 + pad)


def extract_text(page: fitz.Page, rect: fitz.Rect) -> str:
    words = page.get_text("words", clip=rect)
    if words:
        # sort by y then x and join words; this is usually cleaner for tiny clips
        words = sorted(words, key=lambda w: (round(w[1], 1), w[0]))
        return normalize_text(" ".join(w[4] for w in words))
    return normalize_text(page.get_text("text", clip=rect))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.run_dir / "aux_rec.pkl").open("rb") as f:
        aux_rec: dict[str, list[dict[str, Any]]] = pickle.load(f)
    coord_items = json.loads(args.coord_manifest.read_text(encoding="utf-8"))
    coord_by_sample = {item["sample_id"]: item for item in coord_items}

    selected = []
    for filename in sorted(aux_rec):
        sid = unique_sample_id(filename)
        meta = coord_by_sample.get(sid)
        if not meta:
            continue
        if args.skip_header_risk and meta.get("header_risk"):
            continue
        selected.append((filename, sid, meta))
        if args.limit and len(selected) >= args.limit:
            break

    pdf_cache: dict[str, fitz.Document] = {}
    sample_reports = []
    region_rows = []
    aux_rec_pymupdf: dict[str, list[dict[str, Any]]] = {}

    for filename, sid, meta in selected:
        pdf_path = str(meta["pdf"])
        if pdf_path not in pdf_cache:
            pdf_cache[pdf_path] = fitz.open(pdf_path)
        doc = pdf_cache[pdf_path]
        page_no = int(meta["page_no"])
        page = doc[page_no - 1]
        coord = meta["coord"]
        with Image.open(meta["table_png"]) as im:
            crop_size = im.size

        out_items = []
        nonempty = 0
        changed = 0
        for idx, item in enumerate(aux_rec[filename]):
            bbox = as_bbox(item)
            if bbox is None:
                continue
            rect = map_crop_box_to_pymupdf_rect(
                bbox, crop_size, coord, page.rect.height, args.pad_points
            )
            pdf_text = extract_text(page, rect)
            master_text = normalize_text(str(item.get("text", "")))
            if pdf_text:
                nonempty += 1
            if pdf_text and pdf_text != master_text:
                changed += 1
            new_item = dict(item)
            new_item["master_text"] = master_text
            new_item["text"] = pdf_text or master_text
            new_item["pymupdf_text"] = pdf_text
            out_items.append(new_item)
            if idx < 20:
                region_rows.append(
                    {
                        "filename": filename,
                        "sample_id": sid,
                        "region_index": idx,
                        "bbox": bbox,
                        "master_text": master_text,
                        "pymupdf_text": pdf_text,
                        "used_text": new_item["text"],
                    }
                )
        aux_rec_pymupdf[filename] = out_items
        sample_reports.append(
            {
                "filename": filename,
                "sample_id": sid,
                "header_risk": meta.get("header_risk"),
                "regions": len(out_items),
                "pymupdf_nonempty_regions": nonempty,
                "pymupdf_differs_from_master_regions": changed,
                "pymupdf_nonempty_ratio": round(nonempty / max(1, len(out_items)), 3),
                "pymupdf_differs_ratio": round(changed / max(1, len(out_items)), 3),
                "pdf": pdf_path,
                "page_no": page_no,
            }
        )

    for doc in pdf_cache.values():
        doc.close()

    with (args.output_dir / "aux_rec_pymupdf_pilot.pkl").open("wb") as f:
        pickle.dump(aux_rec_pymupdf, f)
    (args.output_dir / "sample_report.json").write_text(
        json.dumps(sample_reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "region_examples.json").write_text(
        json.dumps(region_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_regions = sum(row["regions"] for row in sample_reports)
    total_nonempty = sum(row["pymupdf_nonempty_regions"] for row in sample_reports)
    total_changed = sum(row["pymupdf_differs_from_master_regions"] for row in sample_reports)
    summary = {
        "run_dir": str(args.run_dir),
        "coord_manifest": str(args.coord_manifest),
        "output_dir": str(args.output_dir),
        "samples": len(sample_reports),
        "regions": total_regions,
        "pymupdf_nonempty_regions": total_nonempty,
        "pymupdf_nonempty_ratio": round(total_nonempty / max(1, total_regions), 3),
        "pymupdf_differs_from_master_regions": total_changed,
        "pymupdf_differs_ratio": round(total_changed / max(1, total_regions), 3),
        "pad_points": args.pad_points,
        "skip_header_risk": args.skip_header_risk,
        "aux_rec_pymupdf_pilot": str(args.output_dir / "aux_rec_pymupdf_pilot.pkl"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# PyMuPDF Text Extraction Pilot",
        "",
        "Goal: replace MASTER text with text extracted directly from the source PDF at the same PSENet text-region locations.",
        "",
        f"- Samples: `{summary['samples']}`",
        f"- Regions: `{summary['regions']}`",
        f"- PyMuPDF non-empty regions: `{summary['pymupdf_nonempty_regions']}` ({summary['pymupdf_nonempty_ratio']})",
        f"- Regions where PyMuPDF differs from MASTER: `{summary['pymupdf_differs_from_master_regions']}` ({summary['pymupdf_differs_ratio']})",
        f"- Pad points: `{summary['pad_points']}`",
        "",
        "Files:",
        f"- `summary.json`",
        f"- `sample_report.json`",
        f"- `region_examples.json`",
        f"- `aux_rec_pymupdf_pilot.pkl`",
    ]
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
