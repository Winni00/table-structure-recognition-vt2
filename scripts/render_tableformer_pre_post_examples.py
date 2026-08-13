#!/cluster/home/trinhwin/vt2/docling/.venv/bin/python
from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render TableFormer examples before and after Docling matching/postprocessing."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/tableformer_examples/pre_post",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--filenames-file", type=Path, default=None)
    parser.add_argument(
        "--max-preview-rows",
        type=int,
        default=16,
        help="Maximum HTML rows to draw in each preview panel. Use a large value for full-height examples.",
    )
    parser.add_argument(
        "--max-preview-cols",
        type=int,
        default=8,
        help="Maximum HTML columns to draw in each preview panel.",
    )
    return parser.parse_args()


def load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(24)
FONT_SUB = load_font(16)
FONT_SMALL = load_font(12)


def bbox_from_dict(bbox: dict[str, Any]) -> tuple[float, float, float, float] | None:
    if not bbox:
        return None
    return float(bbox["l"]), float(bbox["t"]), float(bbox["r"]), float(bbox["b"])


def bbox_from_list(bbox: list[Any]) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])


def crop_from_boxes(
    boxes: list[tuple[float, float, float, float]],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    valid = [b for b in boxes if b and b[2] > b[0] and b[3] > b[1]]
    if not valid:
        return 0, 0, image_size[0], image_size[1]
    return (
        max(0, math.floor(min(b[0] for b in valid) - 20)),
        max(0, math.floor(min(b[1] for b in valid) - 20)),
        min(image_size[0], math.ceil(max(b[2] for b in valid) + 20)),
        min(image_size[1], math.ceil(max(b[3] for b in valid) + 20)),
    )


def draw_boxes_panel(
    image: Image.Image,
    crop: tuple[int, int, int, int],
    title: str,
    boxes: list[tuple[float, float, float, float]],
    color: str,
    max_side: int = 760,
) -> Image.Image:
    cropped = image.crop(crop).convert("RGB")
    scale = min(2.5, max(1.0, max_side / max(cropped.width, cropped.height)))
    if scale > 1.01:
        cropped = cropped.resize((int(cropped.width * scale), int(cropped.height * scale)))
    draw = ImageDraw.Draw(cropped)
    ox, oy = crop[0], crop[1]
    for x0, y0, x1, y1 in boxes:
        draw.rectangle(
            ((x0 - ox) * scale, (y0 - oy) * scale, (x1 - ox) * scale, (y1 - oy) * scale),
            outline=color,
            width=2,
        )
    panel = Image.new("RGB", (cropped.width, cropped.height + 48), "white")
    panel.paste(cropped, (0, 48))
    pdraw = ImageDraw.Draw(panel)
    pdraw.text((8, 8), title, font=FONT_SUB, fill="black")
    pdraw.text((8, 28), f"{len(boxes)} boxes", font=FONT_SMALL, fill="gray")
    return panel


def normalize_text(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", "", text or ""))
    return " ".join(text.split())


def html_grid(table_html: str, max_rows: int = 16, max_cols: int = 8) -> tuple[list[list[str]], int, int]:
    soup = BeautifulSoup(table_html or "", "html.parser")
    table = soup.find("table")
    if table is None:
        return [], 0, 0
    rows = table.find_all("tr")
    grid: list[list[str]] = []
    max_col_seen = 0
    occupied: dict[tuple[int, int], str] = {}
    for r, tr in enumerate(rows):
        c = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while (r, c) in occupied:
                c += 1
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            text = normalize_text(cell.decode_contents())
            occupied[(r, c)] = text
            for rr in range(r, r + rowspan):
                for cc in range(c, c + colspan):
                    if (rr, cc) != (r, c):
                        occupied[(rr, cc)] = ""
            c += colspan
            max_col_seen = max(max_col_seen, c)
    for r in range(min(len(rows), max_rows)):
        grid.append([occupied.get((r, c), "") for c in range(min(max_col_seen, max_cols))])
    return grid, len(rows), max_col_seen


def html_cells(
    table_html: str,
    max_rows: int = 16,
    max_cols: int = 8,
) -> tuple[list[dict[str, Any]], int, int, int]:
    soup = BeautifulSoup(table_html or "", "html.parser")
    table = soup.find("table")
    if table is None:
        return [], 0, 0, 0
    rows = table.find_all("tr")
    occupied: set[tuple[int, int]] = set()
    cells: list[dict[str, Any]] = []
    max_col_seen = 0
    for r, tr in enumerate(rows):
        c = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while (r, c) in occupied:
                c += 1
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            text = normalize_text(cell.decode_contents())
            for rr in range(r, r + rowspan):
                for cc in range(c, c + colspan):
                    occupied.add((rr, cc))
            if r < max_rows and c < max_cols:
                visible_rowspan = min(rowspan, max_rows - r)
                visible_colspan = min(colspan, max_cols - c)
                if visible_rowspan > 0 and visible_colspan > 0:
                    cells.append(
                        {
                            "row": r,
                            "col": c,
                            "rowspan": visible_rowspan,
                            "colspan": visible_colspan,
                            "text": text,
                        }
                    )
            c += colspan
            max_col_seen = max(max_col_seen, c)
    return cells, len(rows), max_col_seen, min(len(rows), max_rows)


def render_html_preview(
    title: str,
    table_html: str,
    width: int = 980,
    max_rows: int = 16,
    max_cols: int = 8,
) -> Image.Image:
    cells, num_rows, num_cols, shown_rows = html_cells(
        table_html, max_rows=max_rows, max_cols=max_cols
    )
    shown_cols = min(max(num_cols, 1), max_cols)
    cell_w = max(90, width // shown_cols)
    cell_h = 42
    header_h = 52
    height = header_h + max(1, shown_rows) * cell_h + 10
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title, font=FONT_SUB, fill="black")
    draw.text(
        (8, 30),
        f"HTML grid {num_rows}x{num_cols}; showing {shown_rows} rows. Full tree is used for TEDS.",
        font=FONT_SMALL,
        fill="gray",
    )
    for cell in cells:
        x0 = cell["col"] * cell_w
        y0 = header_h + cell["row"] * cell_h
        x1 = x0 + cell["colspan"] * cell_w
        y1 = y0 + cell["rowspan"] * cell_h
        draw.rectangle((x0, y0, x1, y1), outline="#555555", width=1)
        clipped = cell["text"][:34] + ("..." if len(cell["text"]) > 34 else "")
        draw.text((x0 + 4, y0 + 4), clipped, font=FONT_SMALL, fill="black")
    return panel


def select_prediction_files(run_dir: Path, limit: int, filenames_file: Path | None) -> list[Path]:
    if filenames_file:
        names = [line.strip() for line in filenames_file.read_text().splitlines() if line.strip()]
        out = []
        for name in names:
            stem = Path(name).stem
            path = run_dir / f"{stem}.pred.json"
            if path.exists():
                out.append(path)
        return out[:limit]
    per_table_path = run_dir / "per_table.json"
    if per_table_path.exists():
        rows = json.loads(per_table_path.read_text(encoding="utf-8"))
        # Prefer good-but-not-perfect examples, plus a few weaker ones.
        selected = [
            r for r in rows if 0.85 <= float(r.get("teds_s", 0.0)) < 1.0
        ][: max(0, limit // 2)]
        selected += sorted(rows, key=lambda r: float(r.get("teds_s", 0.0)))[: limit - len(selected)]
        files = []
        seen = set()
        for row in selected:
            stem = row["sample"]
            if stem in seen:
                continue
            path = run_dir / f"{stem}.pred.json"
            if path.exists():
                files.append(path)
                seen.add(stem)
        if files:
            return files[:limit]
    return sorted(run_dir.glob("*.pred.json"))[:limit]


def render_one(pred_path: Path, out_dir: Path, max_preview_rows: int, max_preview_cols: int) -> None:
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    image = Image.open(payload["image_path"]).convert("RGB")
    raw = payload["raw_output"][0]
    details = raw["predict_details"]
    pre_boxes = [bbox_from_list(b) for b in details.get("prediction_bboxes_page", [])]
    pre_boxes = [b for b in pre_boxes if b]
    post_boxes = [bbox_from_dict(cell.get("bbox", {})) for cell in raw.get("tf_responses", [])]
    post_boxes = [b for b in post_boxes if b]
    crop = crop_from_boxes(pre_boxes + post_boxes, image.size)

    pre = draw_boxes_panel(image, crop, "Before matching/postprocessing: raw model cell boxes", pre_boxes, "#7b2cbf")
    post = draw_boxes_panel(image, crop, "After matching/postprocessing: tf_responses with matched content", post_boxes, "#198754")
    gt_html_panel = render_html_preview(
        "Ground truth HTML used for TEDS",
        payload["ground_truth"]["gt_html"],
        max_rows=max_preview_rows,
        max_cols=max_preview_cols,
    )
    pred_html_panel = render_html_preview(
        "Evaluated prediction HTML used for TEDS: native html_seq + matched content",
        payload["normalized_output"][0]["structure"].get("html", ""),
        max_rows=max_preview_rows,
        max_cols=max_preview_cols,
    )

    width = max(pre.width + post.width + 24, gt_html_panel.width, pred_html_panel.width)
    height = 94 + max(pre.height, post.height) + gt_html_panel.height + pred_html_panel.height + 28
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{payload['sample']} | TEDS-S={payload['evaluation_result']['teds_s']:.4f}, "
        f"TEDS={payload['evaluation_result']['teds']:.4f}"
    )
    draw.text((8, 8), title, font=FONT_TITLE, fill="black")
    draw.text(
        (8, 42),
        "Purple = raw model structure boxes. Green = matched content boxes only. TEDS is computed from the evaluated HTML previews below.",
        font=FONT_SMALL,
        fill="gray",
    )
    canvas.paste(pre, (0, 94))
    canvas.paste(post, (pre.width + 24, 94))
    html_y = 94 + max(pre.height, post.height) + 12
    canvas.paste(gt_html_panel, (0, html_y))
    canvas.paste(pred_html_panel, (0, html_y + gt_html_panel.height + 8))
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / f"{payload['sample']}_pre_post.png")


def main() -> None:
    args = parse_args()
    files = select_prediction_files(args.run_dir, args.limit, args.filenames_file)
    out_dir = args.output_dir / args.run_dir.name
    for path in files:
        render_one(path, out_dir, args.max_preview_rows, args.max_preview_cols)
    print(f"Wrote {len(files)} pre/post examples to {out_dir}")


if __name__ == "__main__":
    main()
