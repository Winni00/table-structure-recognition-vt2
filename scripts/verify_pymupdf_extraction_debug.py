"""Create visual and textual evidence for PyMuPDF text extraction.

This verifies whether PSENet boxes from a TFLOP OCR-style run are mapped back
to the correct locations on the original PDF pages.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RUN_DIR = ROOT / "results/tflop_paper_collection_updated_crops_broadbestrot_master_public"
DEFAULT_PYMUPDF_RUN_DIR = ROOT / "results/tflop_paper_collection_updated_crops_broadbestrot_pymupdf_public"
DEFAULT_COORD_MANIFEST = ROOT / "results/paper_collection_updated_crops_coord_check/manifest.json"
DEFAULT_OUT_DIR = ROOT / "results/pymupdf_extraction_verification"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--pymupdf-run-dir", type=Path, default=DEFAULT_PYMUPDF_RUN_DIR)
    p.add_argument("--coord-manifest", type=Path, default=DEFAULT_COORD_MANIFEST)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--random-count", type=int, default=20)
    p.add_argument("--worst-count", type=int, default=30)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--pad-points", type=float, default=1.5)
    p.add_argument("--pdf-zoom", type=float, default=2.0)
    p.add_argument("--max-labels", type=int, default=80)
    return p.parse_args()


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
    if rotation == 0:
        return bbox
    ow, oh = original_size
    rw, rh = rotated_size
    x0, y0, x1, y1 = bbox
    points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    mapped: list[tuple[float, float]] = []
    for x, y in points:
        if rotation == 270:
            mapped.append((y, oh - x))
        elif rotation == 90:
            mapped.append((ow - y, x))
        else:
            raise ValueError(f"Unsupported rotation: {rotation}")
    xs = [min(max(px, 0.0), float(ow)) for px, _ in mapped]
    ys = [min(max(py, 0.0), float(oh)) for _, py in mapped]
    if rotation in (90, 270) and not (abs(rw - oh) <= 2 and abs(rh - ow) <= 2):
        # Kept as metadata concern; do not fail because some source images have
        # tiny render-size differences.
        pass
    return [min(xs), min(ys), max(xs), max(ys)]


def crop_pdf_rect(coord: dict[str, Any], page_height: float) -> fitz.Rect:
    left = float(coord["left"])
    right = float(coord["right"])
    top_bottom_origin = float(coord["top"])
    bottom_bottom_origin = float(coord["bottom"])
    page_top = page_height - top_bottom_origin
    page_bottom = page_height - bottom_bottom_origin
    return fitz.Rect(left, page_top, right, page_bottom)


def map_crop_box_to_pdf_rect(
    bbox_original_crop: list[float],
    original_crop_size: tuple[int, int],
    coord: dict[str, Any],
    page_height: float,
    pad: float,
) -> fitz.Rect:
    crop_w, crop_h = original_crop_size
    x0, y0, x1, y1 = bbox_original_crop
    table_rect = crop_pdf_rect(coord, page_height)
    rx0 = table_rect.x0 + (x0 / crop_w) * table_rect.width
    rx1 = table_rect.x0 + (x1 / crop_w) * table_rect.width
    ry0 = table_rect.y0 + (y0 / crop_h) * table_rect.height
    ry1 = table_rect.y0 + (y1 / crop_h) * table_rect.height
    return fitz.Rect(rx0 - pad, ry0 - pad, rx1 + pad, ry1 + pad)


def rect_to_list(rect: fitz.Rect) -> list[float]:
    return [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]


def words_text(page: fitz.Page, rect: fitz.Rect) -> tuple[str, list[dict[str, Any]]]:
    text_rect = rect * page.derotation_matrix if page.rotation else rect
    raw_words = page.get_text("words", clip=text_rect)
    raw_words = sorted(raw_words, key=lambda w: (round(w[1], 1), w[0]))
    words = [
        {"bbox": [float(w[0]), float(w[1]), float(w[2]), float(w[3])], "text": str(w[4])}
        for w in raw_words
    ]
    return normalize_text(" ".join(w["text"] for w in words)), words


def get_font(size: int = 12) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_rect(draw: ImageDraw.ImageDraw, box: list[float], color: str, width: int = 2) -> None:
    for i in range(width):
        draw.rectangle([box[0] - i, box[1] - i, box[2] + i, box[3] + i], outline=color)


def fit_image_height(img: Image.Image, target_h: int) -> Image.Image:
    if img.height == target_h:
        return img
    scale = target_h / img.height
    return img.resize((max(1, int(img.width * scale)), target_h))


def load_worst_filenames(pymupdf_run_dir: Path, count: int) -> list[str]:
    score_path = pymupdf_run_dir / "gt_canonicalized_rescore/ted_score_output.json"
    rows = json.loads(score_path.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda r: (float(r[5]), float(r[4])))
    return [r[0] for r in rows[:count]]


def main() -> None:
    args = parse_args()
    out = args.output_dir
    visuals_dir = out / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_by_filename = {row["filename"]: row for row in manifest}
    coord_items = json.loads(args.coord_manifest.read_text(encoding="utf-8"))
    coord_by_sample = {row["sample_id"]: row for row in coord_items}
    with (args.run_dir / "aux_rec.pkl").open("rb") as f:
        aux_rec: dict[str, list[dict[str, Any]]] = pickle.load(f)

    all_files = sorted(aux_rec)
    rng = random.Random(args.seed)
    worst_files = load_worst_filenames(args.pymupdf_run_dir, args.worst_count)
    remaining = [f for f in all_files if f not in set(worst_files)]
    random_files = rng.sample(remaining, min(args.random_count, len(remaining)))
    selected = []
    for group, files in [("worst_pymupdf", worst_files), ("random", random_files)]:
        for f in files:
            selected.append((group, f))

    pdf_cache: dict[str, fitz.Document] = {}
    font = get_font(12)
    small_font = get_font(10)
    rows: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []

    for sample_index, (group, filename) in enumerate(selected):
        sid = unique_sample_id(filename)
        coord_meta = coord_by_sample.get(sid)
        run_meta = manifest_by_filename.get(filename, {})
        if not coord_meta:
            continue

        pdf_path = str(coord_meta["pdf"])
        if pdf_path not in pdf_cache:
            pdf_cache[pdf_path] = fitz.open(pdf_path)
        doc = pdf_cache[pdf_path]
        page_no = int(coord_meta["page_no"])
        page = doc[page_no - 1]
        coord = coord_meta["coord"]
        table_rect_pdf = crop_pdf_rect(coord, page.rect.height)
        original_png = Path(run_meta.get("original_table_png") or coord_meta["table_png"])
        crop_png = args.run_dir / "images" / filename
        rotation = int(run_meta.get("applied_rotation") or 0)

        with Image.open(original_png) as im:
            original_size = im.size
        with Image.open(crop_png) as im:
            crop_img = im.convert("RGB")
            rotated_size = im.size

        crop_draw = ImageDraw.Draw(crop_img)

        zoom = float(args.pdf_zoom)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pdf_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf_draw = ImageDraw.Draw(pdf_img)

        table_box_px = [v * zoom for v in rect_to_list(table_rect_pdf)]
        draw_rect(pdf_draw, table_box_px, "blue", 4)
        pdf_draw.text((table_box_px[0] + 4, max(0, table_box_px[1] - 16)), "TargetDomain table crop", fill="blue", font=font)

        empty_count = 0
        region_count = 0
        sim_pm_values = []
        for region_index, item in enumerate(aux_rec[filename]):
            bbox = as_bbox(item)
            if bbox is None:
                continue
            region_count += 1
            bbox_orig = unrotate_bbox(bbox, rotation, original_size, rotated_size)
            rect_pdf = map_crop_box_to_pdf_rect(
                bbox_orig, original_size, coord, page.rect.height, args.pad_points
            )
            text_rect = rect_pdf * page.derotation_matrix if page.rotation else rect_pdf
            text_box = normalize_text(page.get_textbox(text_rect))
            text_words, words = words_text(page, rect_pdf)
            pymupdf_text = text_words or text_box
            master_text = normalize_text(str(item.get("text", "")))
            if not pymupdf_text:
                empty_count += 1
            sim_pm = SequenceMatcher(None, pymupdf_text, master_text).ratio() if (pymupdf_text or master_text) else 1.0
            sim_pm_values.append(sim_pm)

            draw_rect(crop_draw, bbox, "magenta", 2)
            if region_index < args.max_labels:
                crop_draw.text((bbox[0], max(0, bbox[1] - 12)), str(region_index), fill="magenta", font=small_font)

            pdf_box = [v * zoom for v in rect_to_list(rect_pdf)]
            draw_rect(pdf_draw, pdf_box, "red", 2)
            for word in words:
                word_rect = fitz.Rect(word["bbox"])
                if page.rotation:
                    word_rect = word_rect * page.rotation_matrix
                word_box = [v * zoom for v in rect_to_list(word_rect)]
                draw_rect(pdf_draw, word_box, "green", 1)
            if region_index < args.max_labels:
                label = f"{region_index}: {pymupdf_text[:36]}"
                x = min(max(pdf_box[0], 0), max(0, pdf_img.width - 260))
                y = min(max(pdf_box[1] - 13, 0), max(0, pdf_img.height - 14))
                pdf_draw.text((x, y), label, fill="red", font=small_font)

            rows.append(
                {
                    "group": group,
                    "pdf_filename": pdf_path,
                    "page_number": page_no,
                    "table_id": sid,
                    "filename": filename,
                    "region_index": region_index,
                    "pse_box_crop_pixels": bbox,
                    "pse_box_original_crop_pixels": bbox_orig,
                    "crop_rectangle_pdf_page": rect_to_list(table_rect_pdf),
                    "mapped_pymupdf_rectangle": rect_to_list(rect_pdf),
                    "mapped_pymupdf_text_rectangle_unrotated": rect_to_list(text_rect),
                    "scale_x": table_rect_pdf.width / float(original_size[0]),
                    "scale_y": table_rect_pdf.height / float(original_size[1]),
                    "rotation_used": rotation,
                    "pymupdf_text_get_textbox": text_box,
                    "pymupdf_words_text": text_words,
                    "pymupdf_words": words,
                    "master_text": master_text,
                    "gt_text": None,
                    "pymupdf_empty": not bool(pymupdf_text),
                    "similarity_pymupdf_vs_master": sim_pm,
                    "similarity_pymupdf_vs_gt": None,
                    "similarity_master_vs_gt": None,
                }
            )

        target_h = min(max(crop_img.height, 700), 1600)
        crop_vis = fit_image_height(crop_img, target_h)
        pdf_vis = fit_image_height(pdf_img, target_h)
        canvas = Image.new("RGB", (crop_vis.width + pdf_vis.width + 30, target_h + 80), "white")
        draw = ImageDraw.Draw(canvas)
        title = f"{group} | {filename} | page={page_no} | rotation={rotation} | regions={region_count} | PyMuPDF empty={empty_count}"
        draw.text((10, 10), title, fill="black", font=font)
        draw.text(
            (10, 32),
            "Left: PSE boxes. Right: mapped PSE rectangles (red) and extracted PDF word boxes (green).",
            fill="black",
            font=font,
        )
        canvas.paste(crop_vis, (10, 70))
        canvas.paste(pdf_vis, (crop_vis.width + 30, 70))
        safe_name = re.sub(r"[^A-Za-z0-9_.%-]+", "_", filename)
        visual_path = visuals_dir / f"{sample_index:03d}_{group}_{safe_name}.png"
        canvas.save(visual_path)

        sample_summaries.append(
            {
                "group": group,
                "filename": filename,
                "sample_id": sid,
                "pdf": pdf_path,
                "page_number": page_no,
                "rotation_used": rotation,
                "regions": region_count,
                "pymupdf_empty_regions": empty_count,
                "mean_similarity_pymupdf_vs_master": sum(sim_pm_values) / max(1, len(sim_pm_values)),
                "visual": str(visual_path),
            }
        )

    for doc in pdf_cache.values():
        doc.close()

    jsonl_path = out / "pymupdf_box_verification.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = out / "pymupdf_box_verification.csv"
    csv_fields = [
        "group",
        "pdf_filename",
        "page_number",
        "table_id",
        "filename",
        "region_index",
        "pse_box_crop_pixels",
        "crop_rectangle_pdf_page",
        "mapped_pymupdf_rectangle",
        "scale_x",
        "scale_y",
        "rotation_used",
        "pymupdf_text_get_textbox",
        "pymupdf_words_text",
        "master_text",
        "gt_text",
        "pymupdf_empty",
        "similarity_pymupdf_vs_master",
        "similarity_pymupdf_vs_gt",
        "similarity_master_vs_gt",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in csv_fields})

    summary = {
        "output_dir": str(out),
        "run_dir": str(args.run_dir),
        "pymupdf_run_dir": str(args.pymupdf_run_dir),
        "coord_manifest": str(args.coord_manifest),
        "random_count_requested": args.random_count,
        "worst_count_requested": args.worst_count,
        "samples": len(sample_summaries),
        "regions": len(rows),
        "pymupdf_empty_regions": sum(1 for row in rows if row["pymupdf_empty"]),
        "mean_similarity_pymupdf_vs_master": sum(row["similarity_pymupdf_vs_master"] for row in rows) / max(1, len(rows)),
        "sample_summaries": sample_summaries,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "README.md").write_text(
        "# PyMuPDF extraction verification\n\n"
        "This folder verifies whether PSENet crop boxes are mapped back to the correct visible text on the original PDF page.\n\n"
        "Generated evidence:\n"
        "- `visuals/`: side-by-side images. Left = PSE boxes, right = mapped PSE rectangles (red) and extracted PDF word boxes (green).\n"
        "- `pymupdf_box_verification.jsonl`: one row per PSE box with coordinates, extracted text and similarities.\n"
        "- `pymupdf_box_verification.csv`: flat CSV version for inspection.\n"
        "- `summary.json`: aggregate summary and list of visuals.\n\n"
        f"Samples: {len(sample_summaries)}\n\n"
        f"Regions: {len(rows)}\n\n"
        f"PyMuPDF empty regions: {summary['pymupdf_empty_regions']}\n\n"
        f"Mean PyMuPDF-vs-MASTER similarity: {summary['mean_similarity_pymupdf_vs_master']:.4f}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
