"""Feasibility check for reconstructing Paper Collection cell boxes.

This is intentionally diagnostic: it does not create training data yet.
It visualizes whether GT HTML structure can be converted into plausible
cell boxes on the existing table PNGs.
"""

from __future__ import annotations

import argparse
import html
import json
import pickle
import re
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RUN_DIR = ROOT / "results/tflop_paper_collection_ocr_style_707"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "cell_box_reconstruction_feasibility"


@dataclass
class Cell:
    row: int
    col: int
    rowspan: int
    colspan: int
    text: str
    tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-simple", type=int, default=10)
    parser.add_argument("--num-complex", type=int, default=10)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def compact_text(value: str) -> str:
    value = normalize_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def text_similarity(a: str, b: str) -> float:
    a = compact_text(a)
    b = compact_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def parse_html_cells(value: str) -> tuple[list[Cell], int, int, int]:
    soup = BeautifulSoup(value, "lxml")
    table = soup.find("table") or soup
    rows = table.find_all("tr")
    occupied: set[tuple[int, int]] = set()
    cells: list[Cell] = []
    max_col = 0
    span_cells = 0
    for r_idx, tr in enumerate(rows):
        c_idx = 0
        for td in tr.find_all(["td", "th"], recursive=False):
            while (r_idx, c_idx) in occupied:
                c_idx += 1
            rowspan = int(td.get("rowspan") or 1)
            colspan = int(td.get("colspan") or 1)
            if rowspan > 1 or colspan > 1:
                span_cells += 1
            text = normalize_text(td.get_text(" ", strip=True))
            cells.append(Cell(r_idx, c_idx, rowspan, colspan, text, td.name))
            for rr in range(rowspan):
                for cc in range(colspan):
                    occupied.add((r_idx + rr, c_idx + cc))
            c_idx += colspan
            max_col = max(max_col, c_idx)
    return cells, len(rows), max_col, span_cells


def content_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    mask = gray < 245
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        h, w = gray.shape
        return 0, 0, w - 1, h - 1
    pad = 3
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(gray.shape[1] - 1, int(xs.max()) + pad),
        min(gray.shape[0] - 1, int(ys.max()) + pad),
    )


def merge_positions(positions: list[int], gap: int = 4) -> list[int]:
    if not positions:
        return []
    positions = sorted(positions)
    groups = [[positions[0]]]
    for p in positions[1:]:
        if p - groups[-1][-1] <= gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(round(sum(g) / len(g))) for g in groups]


def detect_ruling_lines(gray: np.ndarray) -> tuple[list[int], list[int]]:
    inv = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    h, w = gray.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 18), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 18)))
    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel)
    h_proj = (h_lines > 0).sum(axis=1)
    v_proj = (v_lines > 0).sum(axis=0)
    h_pos = [i for i, count in enumerate(h_proj) if count > max(30, w * 0.18)]
    v_pos = [i for i, count in enumerate(v_proj) if count > max(20, h * 0.18)]
    return merge_positions(h_pos, 5), merge_positions(v_pos, 5)


def kmeans_1d(values: list[float], k: int, rounds: int = 35) -> list[float]:
    if k <= 0:
        return []
    if not values:
        return []
    values_np = np.array(values, dtype=np.float32)
    if len(values) <= k:
        centers = sorted(float(v) for v in values)
        while len(centers) < k:
            centers.append(centers[-1])
        return centers
    qs = np.linspace(0, 100, k + 2)[1:-1]
    centers = np.percentile(values_np, qs)
    for _ in range(rounds):
        dist = np.abs(values_np[:, None] - centers[None, :])
        labels = dist.argmin(axis=1)
        new_centers = np.array(
            [values_np[labels == i].mean() if np.any(labels == i) else centers[i] for i in range(k)],
            dtype=np.float32,
        )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return sorted(float(c) for c in centers)


def boundaries_from_centers(centers: list[float], lo: int, hi: int) -> list[int]:
    if not centers:
        return [lo, hi]
    centers = sorted(centers)
    boundaries = [lo]
    for a, b in zip(centers, centers[1:]):
        boundaries.append(int(round((a + b) / 2)))
    boundaries.append(hi)
    return boundaries


def estimate_boundaries(
    gray: np.ndarray,
    rec_items: list[dict[str, Any]],
    rows: int,
    cols: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    x0, y0, x1, y1 = content_bbox(gray)
    h_lines, v_lines = detect_ruling_lines(gray)
    # Keep only lines in the table content area.
    h_in = [p for p in h_lines if y0 - 3 <= p <= y1 + 3]
    v_in = [p for p in v_lines if x0 - 3 <= p <= x1 + 3]
    meta = {
        "content_bbox": [x0, y0, x1, y1],
        "horizontal_lines": len(h_in),
        "vertical_lines": len(v_in),
        "row_boundary_method": "line" if len(h_in) >= rows + 1 else "ocr_distribution",
        "col_boundary_method": "line" if len(v_in) >= cols + 1 else "ocr_distribution",
    }
    if len(h_in) >= rows + 1:
        row_boundaries = sorted(h_in)[: rows + 1]
    else:
        centers = []
        for item in rec_items:
            bbox = item.get("bbox")
            if bbox is not None and len(bbox) == 4:
                centers.append((float(bbox[1]) + float(bbox[3])) / 2.0)
        row_boundaries = boundaries_from_centers(kmeans_1d(centers, rows), y0, y1)
    if len(v_in) >= cols + 1:
        col_boundaries = sorted(v_in)[: cols + 1]
    else:
        centers = []
        for item in rec_items:
            bbox = item.get("bbox")
            if bbox is not None and len(bbox) == 4:
                centers.append((float(bbox[0]) + float(bbox[2])) / 2.0)
        col_boundaries = boundaries_from_centers(kmeans_1d(centers, cols), x0, x1)
    if len(row_boundaries) != rows + 1:
        row_boundaries = list(np.linspace(y0, y1, rows + 1).round().astype(int))
        meta["row_boundary_method"] = "uniform_fallback"
    if len(col_boundaries) != cols + 1:
        col_boundaries = list(np.linspace(x0, x1, cols + 1).round().astype(int))
        meta["col_boundary_method"] = "uniform_fallback"
    return row_boundaries, col_boundaries, meta


def choose_samples(run_dir: Path, num_simple: int, num_complex: int) -> list[dict[str, Any]]:
    manifest = load_json(run_dir / "manifest.json")
    inference = load_json(run_dir / "full_model_inference.json")
    score_path = run_dir / "report_readable" / "per_table_manifest.json"
    if not score_path.exists():
        score_path = run_dir / "report" / "per_table_manifest.json"
    scores = {row["filename"]: row for row in load_json(score_path)}
    items = []
    for row in manifest:
        filename = row["filename"]
        if filename not in inference:
            continue
        cells, n_rows, n_cols, n_spans = parse_html_cells(inference[filename]["answer_string"])
        if n_rows == 0 or n_cols == 0:
            continue
        score = scores.get(filename, {})
        items.append(
            {
                **row,
                "rows": n_rows,
                "cols": n_cols,
                "cells": len(cells),
                "spans": n_spans,
                "teds": float(score.get("teds", 0.0) or 0.0),
                "teds_s": float(score.get("teds_s", 0.0) or 0.0),
            }
        )
    simple = [
        x
        for x in items
        if x["rows"] <= 12 and x["cols"] <= 8 and x["spans"] <= 2
    ]
    simple = sorted(simple, key=lambda x: (-x["teds_s"], x["rows"] * x["cols"]))[:num_simple]
    complex_items = [
        x
        for x in items
        if x["rows"] >= 10 or x["cols"] >= 10 or x["spans"] >= 4
    ]
    complex_items = sorted(
        complex_items,
        key=lambda x: (-x["spans"], -(x["rows"] * x["cols"]), -x["teds_s"]),
    )[:num_complex]
    selected = []
    seen = set()
    for group, rows in [("simple", simple), ("complex", complex_items)]:
        for item in rows:
            if item["filename"] in seen:
                continue
            seen.add(item["filename"])
            selected.append({**item, "group": group})
    return selected


def text_clip(value: str, limit: int = 28) -> str:
    value = normalize_text(value)
    if len(value) > limit:
        return value[: limit - 1] + "…"
    return value


def draw_example(
    item: dict[str, Any],
    out_path: Path,
    run_dir: Path,
    aux_rec: dict[str, list[dict[str, Any]]],
    inference: dict[str, Any],
) -> dict[str, Any]:
    filename = item["filename"]
    image = Image.open(run_dir / "images" / filename).convert("RGB")
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    rec_items = aux_rec.get(filename, [])
    gt_html = inference[filename]["answer_string"]
    cells, n_rows, n_cols, n_spans = parse_html_cells(gt_html)
    row_b, col_b, meta = estimate_boundaries(gray, rec_items, n_rows, n_cols)
    cell_boxes = []
    for cell in cells:
        if cell.row + cell.rowspan > len(row_b) - 1 or cell.col + cell.colspan > len(col_b) - 1:
            continue
        cell_boxes.append(
            {
                "cell": cell,
                "bbox": [
                    float(col_b[cell.col]),
                    float(row_b[cell.row]),
                    float(col_b[cell.col + cell.colspan]),
                    float(row_b[cell.row + cell.rowspan]),
                ],
                "ocr_hits": 0,
            }
        )
    assigned_ocr = 0
    ocr_assignments: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(cell_boxes))}
    for rec in rec_items:
        bbox = rec.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        for box_idx, box in enumerate(cell_boxes):
            x0, y0, x1, y1 = box["bbox"]
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                box["ocr_hits"] += 1
                ocr_assignments[box_idx].append(rec)
                assigned_ocr += 1
                break
    non_empty_cells = [box for box in cell_boxes if box["cell"].text]
    non_empty_with_ocr = [box for box in non_empty_cells if box["ocr_hits"] > 0]
    similarities = []
    cell_debug = []
    for box_idx, box in enumerate(cell_boxes):
        cell = box["cell"]
        assigned = sorted(
            ocr_assignments[box_idx],
            key=lambda rec: (float(rec["bbox"][1]) if rec.get("bbox") is not None else 0.0,
                             float(rec["bbox"][0]) if rec.get("bbox") is not None else 0.0),
        )
        ocr_text = normalize_text(" ".join(str(rec.get("text", "")) for rec in assigned))
        sim = text_similarity(cell.text, ocr_text)
        if cell.text:
            similarities.append(sim)
        cell_debug.append(
            {
                "row": cell.row,
                "col": cell.col,
                "rowspan": cell.rowspan,
                "colspan": cell.colspan,
                "gt_text": cell.text,
                "ocr_text": ocr_text,
                "ocr_hits": len(assigned),
                "text_similarity": sim,
                "bbox": box["bbox"],
            }
        )
    mean_text_similarity = float(np.mean(similarities)) if similarities else 1.0
    median_text_similarity = float(np.median(similarities)) if similarities else 1.0
    cell_coverage = (len(non_empty_with_ocr) / len(non_empty_cells)) if non_empty_cells else 1.0
    ocr_assignment_rate = (assigned_ocr / len(rec_items)) if rec_items else 1.0
    if (
        cell_coverage >= 0.90
        and ocr_assignment_rate >= 0.95
        and median_text_similarity >= 0.55
        and n_rows <= 25
        and n_cols <= 12
    ):
        quality = "promising"
    elif cell_coverage >= 0.75 and ocr_assignment_rate >= 0.90 and median_text_similarity >= 0.35:
        quality = "manual_check"
    else:
        quality = "weak"

    scale = min(1.0, 1350 / max(image.width, 1))
    display = image.resize((int(image.width * scale), int(image.height * scale)))
    canvas_h = display.height + 170
    canvas_w = max(display.width, 1100)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(display, (0, 110))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        small = ImageFont.truetype("DejaVuSans.ttf", 11)
        bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = small = bold = ImageFont.load_default()

    title = (
        f"{item['group']} | {filename} | rows={n_rows}, cols={n_cols}, spans={n_spans} | "
        f"TEDS-S={item['teds_s']:.4f}, TEDS={item['teds']:.4f}"
    )
    draw.text((8, 8), title, font=bold, fill=(20, 20, 20))
    draw.text(
        (8, 36),
        "Cyan = reconstructed GT cell boxes from GT HTML/grid. Magenta = PSENet+MASTER text-region boxes. Red axes mark image origin (0,0).",
        font=font,
        fill=(70, 70, 70),
    )
    draw.text(
        (8, 62),
        f"Boundary method: rows={meta['row_boundary_method']} ({meta['horizontal_lines']} lines), "
        f"cols={meta['col_boundary_method']} ({meta['vertical_lines']} lines), OCR boxes={len(rec_items)}, "
        f"quality={quality}",
        font=font,
        fill=(70, 70, 70),
    )
    draw.text(
        (8, 86),
        f"Coverage={cell_coverage:.2f}, OCR assigned={ocr_assignment_rate:.2f}, "
        f"median text similarity={median_text_similarity:.2f}",
        font=font,
        fill=(70, 70, 70),
    )
    # origin axes
    ox, oy = 0, 110
    draw.line([(ox, oy), (ox + 45, oy)], fill=(230, 0, 0), width=3)
    draw.line([(ox, oy), (ox, oy + 45)], fill=(230, 0, 0), width=3)
    draw.text((ox + 4, oy + 4), "(0,0)", font=small, fill=(230, 0, 0))

    # OCR boxes.
    for rec in rec_items:
        bbox = rec.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [float(v) * scale for v in bbox]
        draw.rectangle([x0, y0 + 110, x1, y1 + 110], outline=(210, 0, 180), width=1)

    # GT cell boxes.
    max_labels = 80
    for idx, box in enumerate(cell_boxes):
        cell = box["cell"]
        x0, y0_raw, x1, y1_raw = box["bbox"]
        x0 *= scale
        x1 *= scale
        y0 = y0_raw * scale + 110
        y1 = y1_raw * scale + 110
        draw.rectangle([x0, y0, x1, y1], outline=(0, 185, 210), width=2)
        if idx < max_labels and cell.text:
            draw.text((x0 + 2, y0 + 2), text_clip(cell.text), font=small, fill=(0, 120, 145))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "filename": filename,
        "group": item["group"],
        "output_png": str(out_path),
        "rows": n_rows,
        "cols": n_cols,
        "cells": len(cells),
        "spans": n_spans,
        "teds_s": item["teds_s"],
        "teds": item["teds"],
        "ocr_boxes": len(rec_items),
        "assigned_ocr_boxes_by_center": assigned_ocr,
        "non_empty_cells": len(non_empty_cells),
        "non_empty_cells_with_ocr_center": len(non_empty_with_ocr),
        "cell_coverage": cell_coverage,
        "ocr_assignment_rate": ocr_assignment_rate,
        "mean_cell_text_similarity": mean_text_similarity,
        "median_cell_text_similarity": median_text_similarity,
        "quality": quality,
        "cell_debug": cell_debug,
        **meta,
    }


def make_contact_sheet(rows: list[dict[str, Any]], output_dir: Path) -> None:
    for group in sorted({row["group"] for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        thumbs = []
        for row in group_rows:
            img = Image.open(row["output_png"]).convert("RGB")
            img.thumbnail((520, 420))
            tile = Image.new("RGB", (540, 470), "white")
            tile.paste(img, (10, 10))
            draw = ImageDraw.Draw(tile)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 13)
                bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
            except OSError:
                font = bold = ImageFont.load_default()
            draw.text((10, 425), row["quality"], font=bold, fill=(20, 20, 20))
            draw.text(
                (10, 444),
                f"cov={row['cell_coverage']:.2f} sim={row['median_cell_text_similarity']:.2f}",
                font=font,
                fill=(60, 60, 60),
            )
            thumbs.append(tile)
        if not thumbs:
            continue
        cols = 2
        out = Image.new("RGB", (cols * 540, ((len(thumbs) + 1) // cols) * 470), "white")
        for idx, tile in enumerate(thumbs):
            out.paste(tile, ((idx % cols) * 540, (idx // cols) * 470))
        out.save(output_dir / f"contact_sheet_{group}.png")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "aux_rec.pkl").open("rb") as f:
        aux_rec = pickle.load(f)
    inference = load_json(args.run_dir / "full_model_inference.json")
    selected = choose_samples(args.run_dir, args.num_simple, args.num_complex)
    manifest = []
    for idx, item in enumerate(selected, start=1):
        out = args.output_dir / item["group"] / f"{idx:02d}_{item['filename']}"
        manifest.append(draw_example(item, out, args.run_dir, aux_rec, inference))
    (args.output_dir / "cell_assignment_debug.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    compact_manifest = []
    for row in manifest:
        compact = dict(row)
        compact.pop("cell_debug", None)
        compact_manifest.append(compact)
    (args.output_dir / "manifest.json").write_text(json.dumps(compact_manifest, indent=2), encoding="utf-8")
    lines = [
        "# Paper Collection Cell-Box Reconstruction Feasibility",
        "",
        "This is a diagnostic proof of concept, not final annotation data.",
        "",
        "Goal: check whether GT HTML structure can be turned into plausible cell boxes on the existing table PNGs.",
        "",
        "Legend:",
        "",
        "- Cyan: reconstructed GT cell boxes from GT HTML/grid",
        "- Magenta: PSENet+MASTER OCR text-region boxes",
        "- Red axes: image coordinate origin `(0,0)`",
        "",
        "Important: if row/column boundaries are estimated from OCR distribution or uniform fallback, these are approximate boxes, not true annotations.",
        "",
        "## Examples",
        "",
    ]
    for row in compact_manifest:
        rel = Path(row["output_png"]).relative_to(args.output_dir)
        lines.append(
            f"- `{row['group']}` `{row['filename']}`: rows={row['rows']}, cols={row['cols']}, "
            f"spans={row['spans']}, OCR boxes={row['ocr_boxes']}, "
            f"OCR-center assigned={row['assigned_ocr_boxes_by_center']}/{row['ocr_boxes']}, "
            f"non-empty cells with OCR={row['non_empty_cells_with_ocr_center']}/{row['non_empty_cells']}, "
            f"coverage={row['cell_coverage']:.2f}, text-sim-med={row['median_cell_text_similarity']:.2f}, "
            f"quality={row['quality']}, "
            f"row_method={row['row_boundary_method']}, col_method={row['col_boundary_method']}, "
            f"TEDS-S={row['teds_s']:.4f}, TEDS={row['teds']:.4f} -> [{rel}]({rel})"
        )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_contact_sheet(compact_manifest, args.output_dir)
    print(f"wrote {len(manifest)} examples to {args.output_dir}")


if __name__ == "__main__":
    main()
