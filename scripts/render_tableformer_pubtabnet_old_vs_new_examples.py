#!/cluster/home/trinhwin/vt2/docling/.venv/bin/python
from __future__ import annotations

import csv
import html
import importlib.util
import json
import math
import re
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
ORIG_DIR = ROOT / "results/tableformer_pubtabnet_hf_original/val_full"
RECON_DIR = ROOT / "results/tableformer_pubtabnet_hf_bbox_reconstruction/val_full_9115_reconstructed"
PUBTABNET_JSONL = ROOT / "data/pubtabnet_hf/extracted/pubtabnet/PubTabNet_2.0.0.jsonl"
OUT_DIR = ROOT / "results/tableformer_examples/pubtabnet_old_vs_new_adapter_comparisons"
EXPERIMENT_SCRIPT = ROOT / "tableformer_pubtabnet_repro_bbox_experiment.py"


def load_experiment_module():
    spec = importlib.util.spec_from_file_location("tf_bbox_exp", EXPERIMENT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


EXP = load_experiment_module()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(28)
FONT_SUB = load_font(20)
FONT_TEXT = load_font(16)
FONT_SMALL = load_font(14)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", value or ""))).strip()


def bbox_from_cell(cell: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = cell.get("bbox")
    if isinstance(bbox, dict):
        return float(bbox["l"]), float(bbox["t"]), float(bbox["r"]), float(bbox["b"])
    if isinstance(bbox, list) and len(bbox) >= 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return None


def load_pubtabnet_rows_map(requested_filenames: set[str]) -> dict[str, dict[str, Any]]:
    rows = {}
    with PUBTABNET_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            filename = row["filename"]
            if filename in requested_filenames:
                rows[filename] = row
            if len(rows) == len(requested_filenames):
                break
    return rows


def select_examples() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    orig = json.loads((ORIG_DIR / "per_table.json").read_text(encoding="utf-8"))
    recon = json.loads((RECON_DIR / "per_table.json").read_text(encoding="utf-8"))
    recon_map = {row["sample"]: row for row in recon}
    rows = []
    for o in orig:
        r = recon_map.get(o["sample"])
        if not r:
            continue
        rows.append(
            {
                "sample": o["sample"],
                "filename": o["filename"],
                "orig_teds_s": o["teds_s"],
                "orig_teds": o["teds"],
                "recon_teds_s": r["teds_s"],
                "recon_teds": r["teds"],
                "delta_teds_s": r["teds_s"] - o["teds_s"],
                "delta_teds": r["teds"] - o["teds"],
                "reconstructed_empty_bboxes": r.get("cell_bbox_preprocessing", {}).get(
                    "reconstructed_empty_bboxes", 0
                ),
            }
        )

    def html_shape_product(sample: str) -> int:
        pred_path = ORIG_DIR / f"{sample}.pred.json"
        payload = json.loads(pred_path.read_text(encoding="utf-8"))
        cells, num_rows, num_cols = html_table_cells(payload["ground_truth"]["gt_html"])
        return num_rows * num_cols

    high = sorted(
        [r for r in rows if 0.9 <= r["orig_teds_s"] < 1.0 and r["reconstructed_empty_bboxes"] > 0],
        key=lambda x: (x["orig_teds_s"], x["orig_teds"]),
    )
    high_idxs = [0, len(high) // 4, len(high) // 2, (3 * len(high)) // 4, len(high) - 1]
    chosen_high: list[dict[str, Any]] = []
    seen = set()
    for idx in high_idxs:
        row = high[idx]
        if row["sample"] not in seen:
            chosen_high.append(row)
            seen.add(row["sample"])

    # Avoid one huge near-perfect table in the visual set; the HTML panel has
    # to stay readable for discussion slides.
    readable_high = sorted(
        [
            r
            for r in high
            if r["sample"] not in seen
            and r["orig_teds_s"] >= 0.95
            and html_shape_product(r["sample"]) <= 180
        ],
        key=lambda x: (-x["orig_teds_s"], x["orig_teds"]),
    )
    for idx, row in enumerate(list(chosen_high)):
        if html_shape_product(row["sample"]) > 220 and readable_high:
            replacement = readable_high.pop(0)
            seen.discard(row["sample"])
            chosen_high[idx] = replacement
            seen.add(replacement["sample"])

    worse = sorted(
        [r for r in rows if r["delta_teds_s"] < 0 or r["delta_teds"] < 0],
        key=lambda x: (x["delta_teds_s"], x["delta_teds"]),
    )[:5]
    return chosen_high, worse


def draw_multiline_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill, max_width: int):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if not lines:
        lines = [""]
    draw.multiline_text(xy, "\n".join(lines), font=font, fill=fill, spacing=4)
    bbox = draw.multiline_textbbox(xy, "\n".join(lines), font=font, spacing=4)
    return bbox


def crop_box_from_all_boxes(box_sets: list[list[tuple[float, float, float, float]]], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    boxes = [b for group in box_sets for b in group]
    if not boxes:
        return (0, 0, image_size[0], image_size[1])
    x0 = max(0, math.floor(min(b[0] for b in boxes) - 20))
    y0 = max(0, math.floor(min(b[1] for b in boxes) - 20))
    x1 = min(image_size[0], math.ceil(max(b[2] for b in boxes) + 20))
    y1 = min(image_size[1], math.ceil(max(b[3] for b in boxes) + 20))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, image_size[0], image_size[1])
    return (x0, y0, x1, y1)


def make_overlay(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    boxes: list[tuple[float, float, float, float]],
    color: str,
    title: str,
    extra_boxes: list[tuple[tuple[float, float, float, float], str]] | None = None,
) -> Image.Image:
    cropped = image.crop(crop_box).convert("RGB")
    draw = ImageDraw.Draw(cropped)
    ox, oy = crop_box[0], crop_box[1]
    for bbox in boxes:
        x0, y0, x1, y1 = bbox
        draw.rectangle((x0 - ox, y0 - oy, x1 - ox, y1 - oy), outline=color, width=2)
    if extra_boxes:
        for bbox, extra_color in extra_boxes:
            x0, y0, x1, y1 = bbox
            draw.rectangle((x0 - ox, y0 - oy, x1 - ox, y1 - oy), outline=extra_color, width=3)

    panel = Image.new("RGB", (cropped.width, cropped.height + 42), "white")
    panel.paste(cropped, (0, 42))
    pdraw = ImageDraw.Draw(panel)
    pdraw.text((6, 8), title, font=FONT_SUB, fill="black")
    return panel


def html_table_cells(table_html: str) -> tuple[list[dict[str, Any]], int, int]:
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if table is None:
        return [], 0, 0
    rows = table.find_all("tr")
    grid: list[list[bool]] = []
    cells = []
    max_cols = 0
    for row_idx, tr in enumerate(rows):
        while len(grid) <= row_idx:
            grid.append([])
        col_idx = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while col_idx < len(grid[row_idx]) and grid[row_idx][col_idx]:
                col_idx += 1
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            for rr in range(row_idx, row_idx + rowspan):
                while len(grid) <= rr:
                    grid.append([])
                while len(grid[rr]) < col_idx + colspan:
                    grid[rr].append(False)
                for cc in range(col_idx, col_idx + colspan):
                    grid[rr][cc] = True
            cells.append(
                {
                    "start_row": row_idx,
                    "end_row": row_idx + rowspan,
                    "start_col": col_idx,
                    "end_col": col_idx + colspan,
                    "text": normalize_text(cell.decode_contents()),
                }
            )
            max_cols = max(max_cols, col_idx + colspan)
            col_idx += colspan
    return cells, len(rows), max_cols


def render_html_table_panel(
    title: str,
    table_html: str,
    target_width: int = 980,
    max_height: int = 900,
) -> Image.Image:
    cells, num_rows, num_cols = html_table_cells(table_html)
    if num_rows == 0 or num_cols == 0:
        img = Image.new("RGB", (target_width, 120), "white")
        draw = ImageDraw.Draw(img)
        draw.text((12, 12), title, font=FONT_SUB, fill="black")
        draw.text((12, 60), "No HTML table found", font=FONT_TEXT, fill="gray")
        return img

    cell_w = max(8, min(180, target_width // max(1, num_cols)))
    cell_h = max(8, min(42, (max_height - 40) // max(1, num_rows)))
    header_h = 40
    width = cell_w * num_cols + 2
    height = header_h + cell_h * num_rows + 2
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), title, font=FONT_SUB, fill="black")
    compact_mode = cell_w < 22 or cell_h < 18 or (num_rows * num_cols) > 400

    top = header_h
    for cell in cells:
        x0 = cell["start_col"] * cell_w
        y0 = top + cell["start_row"] * cell_h
        x1 = cell["end_col"] * cell_w
        y1 = top + cell["end_row"] * cell_h
        draw.rectangle((x0, y0, x1, y1), outline="black", width=1)
        if not compact_mode:
            text = cell["text"] or " "
            wrapped = textwrap.fill(text, width=max(8, cell_w // 10))
            draw.multiline_text((x0 + 4, y0 + 4), wrapped, font=FONT_SMALL, fill="black", spacing=2)
    if compact_mode:
        note = f"Compact preview only ({num_rows}x{num_cols}, {len(cells)} cells); text omitted due to grid size."
        draw.rectangle((0, 0, width, header_h - 2), fill="white")
        draw.text((8, 8), title, font=FONT_SUB, fill="black")
        draw.text((8, 28), note, font=FONT_SMALL, fill="gray")
    return img


def combine_panels(sample_info: dict[str, Any], image_path: Path, gt_orig_boxes, gt_recon_boxes, recon_only_boxes, old_pred_boxes, new_pred_boxes, gt_html, old_html, new_html) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    crop_box = crop_box_from_all_boxes(
        [gt_orig_boxes, gt_recon_boxes, old_pred_boxes, new_pred_boxes],
        image.size,
    )

    panels = [
        make_overlay(image, crop_box, gt_orig_boxes, "#1f77b4", f"GT original cell boxes ({len(gt_orig_boxes)})"),
        make_overlay(
            image,
            crop_box,
            gt_recon_boxes,
            "#1f77b4",
            f"GT adjusted cell boxes ({len(gt_recon_boxes)})",
            extra_boxes=[(bbox, "#ff8c00") for bbox in recon_only_boxes],
        ),
        make_overlay(
            image,
            crop_box,
            old_pred_boxes,
            "#228b22",
            f"Old adapter predicted/matched boxes ({len(old_pred_boxes)})",
        ),
        make_overlay(
            image,
            crop_box,
            new_pred_boxes,
            "#7b3fb2",
            f"New adapter predicted/matched boxes ({len(new_pred_boxes)})",
        ),
    ]
    top_width = sum(p.width for p in panels) + 30
    top_height = max(p.height for p in panels)
    html_panels = [
        render_html_table_panel("GT HTML used for TEDS (same in both variants)", gt_html),
        render_html_table_panel("Old adapter prediction HTML", old_html),
        render_html_table_panel("New adapter prediction HTML", new_html),
    ]
    bottom_width = sum(p.width for p in html_panels) + 20
    bottom_height = max(p.height for p in html_panels)

    canvas_w = max(top_width, bottom_width) + 40
    canvas_h = 140 + top_height + 30 + bottom_height + 30
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title = sample_info["sample"]
    subtitle = (
        f"Old adapter: TEDS-S={sample_info['orig_teds_s']:.4f}, TEDS={sample_info['orig_teds']:.4f} | "
        f"New adapter: TEDS-S={sample_info['recon_teds_s']:.4f}, TEDS={sample_info['recon_teds']:.4f}"
    )
    delta = (
        f"Delta: TEDS-S={sample_info['delta_teds_s']:+.4f}, TEDS={sample_info['delta_teds']:+.4f} | "
        f"reconstructed empty-cell bboxes={sample_info['reconstructed_empty_bboxes']}"
    )
    draw.text((20, 16), title, font=FONT_TITLE, fill="black")
    draw.text((20, 54), subtitle, font=FONT_SUB, fill="black")
    draw.text((20, 82), delta, font=FONT_TEXT, fill="black")
    draw.text((20, 106), "Orange boxes in GT adjusted = newly reconstructed empty-cell boxes.", font=FONT_SMALL, fill="gray")

    x = 20
    y = 140
    for panel in panels:
        canvas.paste(panel, (x, y))
        x += panel.width + 10

    x = 20
    y = 140 + top_height + 30
    for panel in html_panels:
        canvas.paste(panel, (x, y))
        x += panel.width + 10
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    high, worse = select_examples()
    requested_filenames = {item["filename"] for item in high + worse}
    print(f"Loading {len(requested_filenames)} requested PubTabNet rows from {PUBTABNET_JSONL}", flush=True)
    rows_map = load_pubtabnet_rows_map(requested_filenames)
    print(f"Selected {len(high)} high_not_one and {len(worse)} worse_after_reconstruction samples", flush=True)
    groups = {
        "high_not_one": high,
        "worse_after_reconstruction": worse,
    }

    summary_rows = []
    for group_name, items in groups.items():
        print(f"Processing group: {group_name}", flush=True)
        group_dir = OUT_DIR / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            filename = item["filename"]
            sample = item["sample"]
            print(f"  Sample: {sample}", flush=True)
            row = deepcopy(rows_map[filename])

            orig_pred = json.loads((ORIG_DIR / f"{sample}.pred.json").read_text(encoding="utf-8"))
            recon_pred = json.loads((RECON_DIR / f"{sample}.pred.json").read_text(encoding="utf-8"))

            image_path = Path(orig_pred["image_path"])
            image = Image.open(image_path)
            width, height = image.size
            image.close()

            orig_gt_cells = [c for c in deepcopy(row["html"]["cells"]) if len(c.get("bbox", [])) >= 4]
            prepared_recon, recon_stats = EXP.reconstruct_missing_cell_bboxes(deepcopy(row), width, height)
            recon_gt_cells = prepared_recon["html"]["cells"]

            gt_orig_boxes = [bbox_from_cell(c) for c in orig_gt_cells]
            gt_orig_boxes = [b for b in gt_orig_boxes if b]
            gt_recon_boxes = [bbox_from_cell(c) for c in recon_gt_cells]
            gt_recon_boxes = [b for b in gt_recon_boxes if b]
            recon_only_boxes = [bbox_from_cell(c) for c in recon_gt_cells if c.get("bbox_reconstructed")]
            recon_only_boxes = [b for b in recon_only_boxes if b]

            old_pred_boxes = [bbox_from_cell(c) for c in orig_pred["normalized_output"][0]["cells"]]
            old_pred_boxes = [b for b in old_pred_boxes if b]
            new_pred_boxes = [bbox_from_cell(c) for c in recon_pred["normalized_output"][0]["cells"]]
            new_pred_boxes = [b for b in new_pred_boxes if b]

            gt_html = orig_pred["ground_truth"]["gt_html"]
            old_html = orig_pred["normalized_output"][0]["structure"]["html"]
            new_html = recon_pred["normalized_output"][0]["structure"]["html"]

            canvas = combine_panels(
                item,
                image_path,
                gt_orig_boxes,
                gt_recon_boxes,
                recon_only_boxes,
                old_pred_boxes,
                new_pred_boxes,
                gt_html,
                old_html,
                new_html,
            )
            out_path = group_dir / f"{sample}_comparison.png"
            canvas.save(out_path)
            print(f"    Saved: {out_path}", flush=True)

            summary_rows.append(
                {
                    **item,
                    "group": group_name,
                    "recon_gt_total_boxes": len(gt_recon_boxes),
                    "orig_gt_total_boxes": len(gt_orig_boxes),
                    "old_pred_boxes": len(old_pred_boxes),
                    "new_pred_boxes": len(new_pred_boxes),
                    "output_path": str(out_path),
                    "reconstruction_stats": recon_stats,
                }
            )

    write_json(OUT_DIR / "summary.json", summary_rows)
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "sample",
                "filename",
                "orig_teds_s",
                "orig_teds",
                "recon_teds_s",
                "recon_teds",
                "delta_teds_s",
                "delta_teds",
                "reconstructed_empty_bboxes",
                "orig_gt_total_boxes",
                "recon_gt_total_boxes",
                "old_pred_boxes",
                "new_pred_boxes",
                "output_path",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            slim = {k: row[k] for k in writer.fieldnames}
            writer.writerow(slim)
    print(f"Wrote {len(summary_rows)} comparisons to {OUT_DIR}")


if __name__ == "__main__":
    main()
