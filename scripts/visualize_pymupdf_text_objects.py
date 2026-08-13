"""Visualize PDF text objects around mapped PSENet regions.

The existing PyMuPDF verification proves that crop-to-PDF mapping is broadly
correct. This script adds the missing diagnostic layer: full PDF spans and
characters intersecting selected mapped PSE boxes, including object extents
outside the extraction rectangle.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import traceback
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont

from verify_pymupdf_extraction_debug import (
    ROOT,
    as_bbox,
    crop_pdf_rect,
    get_font,
    map_crop_box_to_pdf_rect,
    normalize_text,
    unique_sample_id,
    unrotate_bbox,
    words_text,
)


DEFAULT_RUN_DIR = ROOT / "results/tflop_paper_collection_updated_crops_broadbestrot_master_public"
DEFAULT_PYMUPDF_RUN_DIR = ROOT / "results/tflop_paper_collection_updated_crops_broadbestrot_pymupdf_public"
DEFAULT_COORD_MANIFEST = ROOT / "results/paper_collection_updated_crops_coord_check/manifest.json"
DEFAULT_OUT_DIR = ROOT / "results/pymupdf_text_object_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--pymupdf-run-dir", type=Path, default=DEFAULT_PYMUPDF_RUN_DIR)
    parser.add_argument("--coord-manifest", type=Path, default=DEFAULT_COORD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--worst-count", type=int, default=10)
    parser.add_argument("--good-count", type=int, default=5)
    parser.add_argument("--regions-per-table", type=int, default=6)
    parser.add_argument("--pad-points", type=float, default=1.5)
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--filename", help="Render one exact filename instead of score-based samples")
    return parser.parse_args()


def rect_list(rect: fitz.Rect) -> list[float]:
    return [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]


def intersects(a: fitz.Rect, b: fitz.Rect) -> bool:
    return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0)


def intersection_fraction(obj: fitz.Rect, region: fitz.Rect) -> float:
    overlap = obj & region
    if overlap.is_empty or obj.get_area() <= 0:
        return 0.0
    return overlap.get_area() / obj.get_area()


def flatten_rawdict(page: fitz.Page) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return full-page spans and characters in PyMuPDF's stored order."""
    raw = page.get_text("rawdict")
    spans: list[dict[str, Any]] = []
    chars: list[dict[str, Any]] = []
    order = 0
    for block_index, block in enumerate(raw.get("blocks", [])):
        for line_index, line in enumerate(block.get("lines", [])):
            for span_index, span in enumerate(line.get("spans", [])):
                span_chars = span.get("chars", [])
                text = "".join(str(ch.get("c", "")) for ch in span_chars)
                span_row = {
                    "order": order,
                    "block": block_index,
                    "line": line_index,
                    "span": span_index,
                    "bbox": [float(v) for v in span["bbox"]],
                    "text": text,
                    "font": span.get("font"),
                    "size": span.get("size"),
                }
                spans.append(span_row)
                for char_index, char in enumerate(span_chars):
                    chars.append(
                        {
                            "order": len(chars),
                            "span_order": order,
                            "char": char_index,
                            "bbox": [float(v) for v in char["bbox"]],
                            "text": str(char.get("c", "")),
                        }
                    )
                order += 1
    return spans, chars


def select_scores(run_dir: Path, worst_count: int, good_count: int) -> list[tuple[str, str]]:
    score_path = run_dir / "gt_canonicalized_rescore/ted_score_output.json"
    rows = json.loads(score_path.read_text(encoding="utf-8"))
    ordered = sorted(rows, key=lambda row: (float(row[5]), float(row[4])))
    worst = [("worst_pymupdf", row[0]) for row in ordered[:worst_count]]
    good_pool = [row for row in reversed(ordered) if float(row[5]) >= 0.9]
    good = [("good_pymupdf", row[0]) for row in good_pool[:good_count]]
    if len(good) < good_count:
        good = [("good_pymupdf", row[0]) for row in reversed(ordered[:])[:good_count]]
    return worst + good


def display_rect(rect: fitz.Rect, page: fitz.Page, clip: fitz.Rect, zoom: float) -> list[float]:
    shown = rect * page.rotation_matrix if page.rotation else rect
    shown_clip = clip * page.rotation_matrix if page.rotation else clip
    return [
        (shown.x0 - shown_clip.x0) * zoom,
        (shown.y0 - shown_clip.y0) * zoom,
        (shown.x1 - shown_clip.x0) * zoom,
        (shown.y1 - shown_clip.y0) * zoom,
    ]


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: str, width: int = 2) -> None:
    for offset in range(width):
        draw.rectangle(
            [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
            outline=color,
        )


def wrap(text: str, width: int = 86) -> list[str]:
    text = normalize_text(text)
    if not text:
        return ["<empty>"]
    return [text[i : i + width] for i in range(0, len(text), width)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = args.output_dir / "visuals"
    visuals_dir.mkdir(exist_ok=True)

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_by_file = {row["filename"]: row for row in manifest}
    coords = json.loads(args.coord_manifest.read_text(encoding="utf-8"))
    coords_by_id = {row["sample_id"]: row for row in coords}
    with (args.run_dir / "aux_rec.pkl").open("rb") as handle:
        master: dict[str, list[dict[str, Any]]] = pickle.load(handle)

    selected = (
        [("manual", args.filename)]
        if args.filename
        else select_scores(args.pymupdf_run_dir, args.worst_count, args.good_count)
    )
    random.Random(args.seed).shuffle(selected[: args.worst_count])
    font = get_font(17)
    small = get_font(13)
    records: list[dict[str, Any]] = []

    failures: list[dict[str, str]] = []
    for sample_no, (group, filename) in enumerate(selected):
        try:
            record = process_sample(
                args, sample_no, group, filename, manifest_by_file, coords_by_id, master, font, small, visuals_dir
            )
            if record:
                records.append(record)
            print(f"[{sample_no + 1}/{len(selected)}] {filename}", flush=True)
        except Exception as exc:  # Keep one malformed PDF from losing the report.
            failures.append({"group": group, "filename": filename, "error": repr(exc)})
            print(f"FAILED {filename}: {exc}", flush=True)
            traceback.print_exc()

    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# PyMuPDF PDF-text-object diagnostics\n\n"
        "This report overlays mapped PSENet extraction rectangles with the full PDF text objects that intersect them.\n\n"
        "- Thick colored boxes: mapped PSE regions used for extraction.\n"
        "- Cyan boxes: complete PDF spans, including parts outside a PSE region.\n"
        "- Orange boxes: individual PDF character objects.\n"
        "- Side panel: PyMuPDF text, MASTER text, stored span order, and partial-span count.\n\n"
        f"Generated tables: {len(records)} ({args.worst_count} poor + {args.good_count} good requested).\n\n"
        f"Skipped failures: {len(failures)}.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "tables": len(records), "failures": len(failures)}, indent=2))


def process_sample(
    args: argparse.Namespace,
    sample_no: int,
    group: str,
    filename: str,
    manifest_by_file: dict[str, dict[str, Any]],
    coords_by_id: dict[str, dict[str, Any]],
    master: dict[str, list[dict[str, Any]]],
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
    visuals_dir: Path,
) -> dict[str, Any] | None:
        sid = unique_sample_id(filename)
        coord_meta = coords_by_id.get(sid)
        run_meta = manifest_by_file.get(filename)
        if not coord_meta or not run_meta or filename not in master:
            return None

        doc = fitz.open(coord_meta["pdf"])
        page = doc[int(coord_meta["page_no"]) - 1]
        spans, chars = flatten_rawdict(page)
        coord = coord_meta["coord"]
        table_rect = crop_pdf_rect(coord, page.rect.height)
        raw_table_rect = table_rect * page.derotation_matrix if page.rotation else table_rect
        # Keep only PDF objects near the table. Dense scientific PDFs may have
        # tens of thousands of characters on the page; comparing every PSE
        # region against the full page is needlessly quadratic.
        object_margin = 6.0
        raw_table_rect = fitz.Rect(
            raw_table_rect.x0 - object_margin,
            raw_table_rect.y0 - object_margin,
            raw_table_rect.x1 + object_margin,
            raw_table_rect.y1 + object_margin,
        )
        spans = [row for row in spans if intersects(fitz.Rect(row["bbox"]), raw_table_rect)]
        chars = [row for row in chars if intersects(fitz.Rect(row["bbox"]), raw_table_rect)]
        original_png = Path(run_meta.get("original_table_png") or coord_meta["table_png"])
        crop_png = args.run_dir / "images" / filename
        with Image.open(original_png) as image:
            original_size = image.size
        with Image.open(crop_png) as image:
            rotated_size = image.size
        rotation = int(run_meta.get("applied_rotation") or 0)

        candidates: list[dict[str, Any]] = []
        for region_index, item in enumerate(master[filename]):
            bbox = as_bbox(item)
            if bbox is None:
                continue
            bbox_original = unrotate_bbox(bbox, rotation, original_size, rotated_size)
            rect = map_crop_box_to_pdf_rect(
                bbox_original, original_size, coord, page.rect.height, args.pad_points
            )
            text_rect = rect * page.derotation_matrix if page.rotation else rect
            textbox = normalize_text(page.get_textbox(text_rect))
            words_value, words = words_text(page, rect)
            pdf_text = words_value or textbox
            master_text = normalize_text(str(item.get("text", "")))
            hit_spans = []
            for span in spans:
                span_rect = fitz.Rect(span["bbox"])
                if intersects(span_rect, text_rect):
                    hit_spans.append({**span, "overlap_fraction": intersection_fraction(span_rect, text_rect)})
            hit_chars = []
            for char in chars:
                char_rect = fitz.Rect(char["bbox"])
                if intersects(char_rect, text_rect):
                    hit_chars.append({**char, "overlap_fraction": intersection_fraction(char_rect, text_rect)})

            partial_spans = sum(1 for span in hit_spans if span["overlap_fraction"] < 0.98)
            similarity = SequenceMatcher(None, pdf_text, master_text).ratio()
            diagnostic_priority = (1.0 - similarity) + min(partial_spans, 3) * 0.25 + min(len(hit_spans), 3) * 0.05
            candidates.append(
                {
                    "region_index": region_index,
                    "pse_box_crop": bbox,
                    "mapped_rect": rect_to_json(rect),
                    "text_rect": rect_to_json(text_rect),
                    "pymupdf_textbox": textbox,
                    "pymupdf_words_text": words_value,
                    "pymupdf_words": words,
                    "master_text": master_text,
                    "similarity": similarity,
                    "spans": hit_spans,
                    "chars": hit_chars,
                    "partial_spans": partial_spans,
                    "priority": diagnostic_priority,
                }
            )

        chosen = sorted(candidates, key=lambda row: row["priority"], reverse=True)[: args.regions_per_table]
        if not chosen:
            doc.close()
            return None

        selected_rects = [fitz.Rect(row["text_rect"]) for row in chosen]
        union = fitz.Rect(selected_rects[0])
        for rect in selected_rects[1:]:
            union |= rect
        margin = max(8.0, min(table_rect.width, table_rect.height) * 0.025)
        clip = fitz.Rect(union.x0 - margin, union.y0 - margin, union.x1 + margin, union.y1 + margin) & page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(args.zoom, args.zoom), clip=clip, alpha=False)
        page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(page_img)

        colors = ["#e31a1c", "#ff7f00", "#6a3d9a", "#b15928", "#1f78b4", "#33a02c"]
        for chosen_index, row in enumerate(chosen):
            region_rect = fitz.Rect(row["text_rect"])
            color = colors[chosen_index % len(colors)]
            draw_box(draw, display_rect(region_rect, page, clip, args.zoom), color, 4)
            for span in row["spans"]:
                draw_box(draw, display_rect(fitz.Rect(span["bbox"]), page, clip, args.zoom), "#00a6d6", 2)
            for char in row["chars"]:
                draw_box(draw, display_rect(fitz.Rect(char["bbox"]), page, clip, args.zoom), "#fdbf6f", 1)
            box = display_rect(region_rect, page, clip, args.zoom)
            draw.text((box[0] + 2, max(0, box[1] - 18)), f"R{row['region_index']}", fill=color, font=small)

        panel_width = 920
        line_height = 19
        panel_lines: list[tuple[str, str]] = [
            ("black", f"{group} | {filename} | page={coord_meta['page_no']} | rotation={rotation}"),
            ("black", "Legend: thick colored=PSE region | cyan=PDF span | orange=PDF character"),
        ]
        for row in chosen:
            span_text = " | ".join(f"#{span['order']}:{normalize_text(span['text'])}" for span in row["spans"])
            panel_lines.append(("#8b0000", f"Region {row['region_index']}  sim(PDF,MASTER)={row['similarity']:.2f}  partial spans={row['partial_spans']}"))
            for label, value in [
                ("PyMuPDF", row["pymupdf_words_text"] or row["pymupdf_textbox"]),
                ("MASTER", row["master_text"]),
                ("PDF spans", span_text),
            ]:
                wrapped = wrap(f"{label}: {value}")
                panel_lines.extend(("black", line) for line in wrapped)

        panel_height = max(page_img.height, 30 + len(panel_lines) * line_height)
        canvas = Image.new("RGB", (page_img.width + panel_width, panel_height), "white")
        canvas.paste(page_img, (0, 0))
        panel_draw = ImageDraw.Draw(canvas)
        y = 12
        for color, line in panel_lines:
            panel_draw.text((page_img.width + 16, y), line, fill=color, font=small)
            y += line_height

        safe = re.sub(r"[^A-Za-z0-9_.%-]+", "_", filename)
        visual = visuals_dir / f"{sample_no:02d}_{group}_{safe}.png"
        canvas.save(visual)
        record = {
            "group": group,
            "filename": filename,
            "pdf": coord_meta["pdf"],
            "page": coord_meta["page_no"],
            "rotation": rotation,
            "visual": str(visual),
            "regions": chosen,
        }
        doc.close()
        return record


def rect_to_json(rect: fitz.Rect) -> list[float]:
    return [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]


if __name__ == "__main__":
    main()
