"""Replace OCR-recognized text with PyMuPDF text for a TFLOP run directory.

The input run directory must already contain:
- `aux.json`
- `aux_rec.pkl` from PSENet+MASTER
- `manifest.json` with optional `applied_rotation`

If images were rotated before OCR, PSENet boxes are in rotated crop coordinates.
This script maps those boxes back to the original crop coordinate system before
mapping them onto the source PDF page.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Any

import fitz
from PIL import Image


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RUN_DIR = ROOT / "results/tflop_paper_collection_updated_crops_bestrot_master_public"
DEFAULT_COORD_MANIFEST = ROOT / "results/paper_collection_updated_crops_coord_check/manifest.json"
DEFAULT_OUT_DIR = ROOT / "results/tflop_paper_collection_updated_crops_bestrot_pymupdf_public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--coord-manifest", type=Path, default=DEFAULT_COORD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pad-points", type=float, default=1.5)
    fallback = parser.add_mutually_exclusive_group()
    fallback.add_argument("--fallback-to-master", dest="fallback_to_master", action="store_true")
    fallback.add_argument("--no-fallback-to-master", dest="fallback_to_master", action="store_false")
    parser.set_defaults(fallback_to_master=True)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def unique_sample_id(filename: str) -> str:
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


def unrotate_bbox(
    bbox: list[float],
    rotation: int,
    original_size: tuple[int, int],
    rotated_size: tuple[int, int],
) -> list[float]:
    """Map a bbox from rotated image coordinates back to original crop pixels."""
    if rotation == 0:
        return bbox
    ow, oh = original_size
    rw, rh = rotated_size
    x0, y0, x1, y1 = bbox
    points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    mapped: list[tuple[float, float]] = []
    for x, y in points:
        if rotation == 270:
            # PIL ROTATE_270 is visually a 90 degree clockwise/right rotation:
            # original(x,y) -> rotated(oh-y, x), so inverse is (y, oh-x).
            mapped.append((y, oh - x))
        elif rotation == 90:
            # PIL ROTATE_90 is visually a 90 degree counter-clockwise/left rotation:
            # original(x,y) -> rotated(y, ow-x), so inverse is (ow-y, x).
            mapped.append((ow - y, x))
        else:
            raise ValueError(f"Unsupported rotation: {rotation}")
    xs = [min(max(px, 0.0), float(ow)) for px, _ in mapped]
    ys = [min(max(py, 0.0), float(oh)) for _, py in mapped]
    # Sanity check dimensions, but do not fail on off-by-one renderer behavior.
    if rotation in (90, 270) and not (abs(rw - oh) <= 2 and abs(rh - ow) <= 2):
        pass
    return [min(xs), min(ys), max(xs), max(ys)]


def map_crop_box_to_pymupdf_rect(
    bbox_original_crop: list[float],
    original_crop_size: tuple[int, int],
    coord: dict[str, Any],
    page_height: float,
    pad: float,
) -> fitz.Rect:
    crop_w, crop_h = original_crop_size
    x0, y0, x1, y1 = bbox_original_crop
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
    # Crop coordinates refer to the visually rotated page, while PyMuPDF text
    # coordinates are returned in the unrotated page coordinate system.
    text_rect = rect * page.derotation_matrix if page.rotation else rect
    words = page.get_text("words", clip=text_rect)
    if words:
        words = sorted(words, key=lambda w: (round(w[1], 1), w[0]))
        return normalize_text(" ".join(w[4] for w in words))
    return normalize_text(page.get_text("text", clip=text_rect))


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.run_dir / "aux.json", out / "aux.json")
    shutil.copy2(args.run_dir / "subset.txt", out / "subset.txt")
    shutil.copy2(args.run_dir / "manifest.json", out / "manifest.json")
    image_link = out / "images"
    if image_link.exists() or image_link.is_symlink():
        image_link.unlink() if image_link.is_symlink() or image_link.is_file() else shutil.rmtree(image_link)
    image_link.symlink_to((args.run_dir / "images").resolve())

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_by_filename = {row["filename"]: row for row in manifest}
    coord_items = json.loads(args.coord_manifest.read_text(encoding="utf-8"))
    coord_by_sample = {row["sample_id"]: row for row in coord_items}
    with (args.run_dir / "aux_rec.pkl").open("rb") as f:
        aux_rec: dict[str, list[dict[str, Any]]] = pickle.load(f)

    pdf_cache: dict[str, fitz.Document] = {}
    aux_rec_pdf: dict[str, list[dict[str, Any]]] = {}
    reports = []
    missing_meta = []
    for filename in sorted(aux_rec):
        sid = unique_sample_id(filename)
        coord_meta = coord_by_sample.get(sid)
        run_meta = manifest_by_filename.get(filename, {})
        if not coord_meta:
            missing_meta.append(filename)
            aux_rec_pdf[filename] = aux_rec[filename]
            continue

        pdf_path = str(coord_meta["pdf"])
        if pdf_path not in pdf_cache:
            pdf_cache[pdf_path] = fitz.open(pdf_path)
        doc = pdf_cache[pdf_path]
        page = doc[int(coord_meta["page_no"]) - 1]
        coord = coord_meta["coord"]
        original_png = Path(run_meta.get("original_table_png") or coord_meta["table_png"])
        with Image.open(original_png) as im:
            original_size = im.size
        with Image.open(args.run_dir / "images" / filename) as im:
            rotated_size = im.size
        rotation = int(run_meta.get("applied_rotation") or 0)

        out_items = []
        nonempty = 0
        changed = 0
        for item in aux_rec[filename]:
            bbox = as_bbox(item)
            if bbox is None:
                continue
            bbox_orig = unrotate_bbox(bbox, rotation, original_size, rotated_size)
            rect = map_crop_box_to_pymupdf_rect(
                bbox_orig, original_size, coord, page.rect.height, args.pad_points
            )
            pdf_text = extract_text(page, rect)
            master_text = normalize_text(str(item.get("text", "")))
            used_text = pdf_text if pdf_text or not args.fallback_to_master else master_text
            if pdf_text:
                nonempty += 1
            if pdf_text and pdf_text != master_text:
                changed += 1
            new_item = dict(item)
            new_item["master_text"] = master_text
            new_item["pymupdf_text"] = pdf_text
            new_item["text"] = used_text
            new_item["bbox_original_crop_for_pymupdf"] = bbox_orig
            new_item["applied_rotation"] = rotation
            out_items.append(new_item)
        aux_rec_pdf[filename] = out_items
        reports.append(
            {
                "filename": filename,
                "sample_id": sid,
                "applied_rotation": rotation,
                "regions": len(out_items),
                "pymupdf_nonempty_regions": nonempty,
                "pymupdf_nonempty_ratio": round(nonempty / max(1, len(out_items)), 3),
                "pymupdf_differs_from_master_regions": changed,
                "pymupdf_differs_ratio": round(changed / max(1, len(out_items)), 3),
            }
        )

    for doc in pdf_cache.values():
        doc.close()

    with (out / "aux_rec.pkl").open("wb") as f:
        pickle.dump(aux_rec_pdf, f)
    (out / "pymupdf_text_report.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    total_regions = sum(row["regions"] for row in reports)
    total_nonempty = sum(row["pymupdf_nonempty_regions"] for row in reports)
    total_changed = sum(row["pymupdf_differs_from_master_regions"] for row in reports)
    summary = {
        "source_run_dir": str(args.run_dir),
        "coord_manifest": str(args.coord_manifest),
        "output_dir": str(out),
        "samples": len(reports),
        "missing_meta": len(missing_meta),
        "regions": total_regions,
        "pymupdf_nonempty_regions": total_nonempty,
        "pymupdf_nonempty_ratio": round(total_nonempty / max(1, total_regions), 3),
        "pymupdf_differs_from_master_regions": total_changed,
        "pymupdf_differs_ratio": round(total_changed / max(1, total_regions), 3),
        "rotated_samples": sum(1 for row in reports if row["applied_rotation"] != 0),
        "pad_points": args.pad_points,
        "fallback_to_master": args.fallback_to_master,
    }
    (out / "pymupdf_text_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
