"""Create a compact report for the paper-collection OCR-style TFLOP run."""

from __future__ import annotations

import argparse
import html
import json
import pickle
import re
import textwrap
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RUN_DIR = ROOT / "results/tflop_paper_collection_ocr_style_707"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--examples-per-group", type=int, default=5)
    parser.add_argument(
        "--box-legend",
        default="Magenta = PSENet+MASTER OCR text-region boxes",
        help="Legend text for the magenta input boxes in example visualizations.",
    )
    parser.add_argument(
        "--pipeline-label",
        default="table PNG -> PSENet text-region detection -> MASTER text recognition -> TFLOP inference -> official TFLOP TEDS/TEDS-S.",
        help="Short pipeline description shown in README.md.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_tags(value: str, limit: int = 1600) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit]


def safe_score(value: float) -> str:
    return f"{value:.4f}"


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, width_chars: int, fill):
    x, y = xy
    for line in textwrap.wrap(text, width=width_chars):
        draw.text((x, y), line, font=font, fill=fill)
        y += 16
    return y


def normalize_cell_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def cell_signature(cell) -> str:
    text = normalize_cell_text(cell.get_text(" ", strip=True))
    attrs = []
    if cell.get("rowspan"):
        attrs.append(f"rowspan={cell.get('rowspan')}")
    if cell.get("colspan"):
        attrs.append(f"colspan={cell.get('colspan')}")
    if attrs:
        return f"{text} [{' '.join(attrs)}]"
    return text


def html_to_grid(value: str, max_rows: int = 40, max_cols: int = 24) -> tuple[list[list[str]], int, int, int]:
    soup = BeautifulSoup(value, "lxml")
    rows = soup.find_all("tr")
    expanded: list[list[str]] = []
    occupied: dict[tuple[int, int], str] = {}
    total_cells = 0
    for r_idx, tr in enumerate(rows):
        while len(expanded) <= r_idx:
            expanded.append([])
        row = expanded[r_idx]
        cells = tr.find_all(["td", "th"], recursive=False)
        total_cells += len(cells)
        c_idx = 0
        for cell in cells:
            while (r_idx, c_idx) in occupied:
                if len(row) <= c_idx:
                    row.extend([""] * (c_idx + 1 - len(row)))
                row[c_idx] = occupied[(r_idx, c_idx)]
                c_idx += 1
            text = cell_signature(cell)
            rowspan = int(cell.get("rowspan") or 1)
            colspan = int(cell.get("colspan") or 1)
            if colspan > 1 or rowspan > 1:
                text = f"{text} [{rowspan}x{colspan}]"
            for rr in range(rowspan):
                while len(expanded) <= r_idx + rr:
                    expanded.append([])
                target = expanded[r_idx + rr]
                for cc in range(colspan):
                    pos = c_idx + cc
                    while len(target) <= pos:
                        target.append("")
                    target[pos] = text if rr == 0 and cc == 0 else "↳"
                    if rr > 0:
                        occupied[(r_idx + rr, pos)] = "↳"
            c_idx += colspan
        for pos, value_at_pos in list(occupied.items()):
            rr, cc = pos
            if rr == r_idx and cc >= len(row):
                row.extend([""] * (cc + 1 - len(row)))
                row[cc] = value_at_pos
    n_rows = len(rows)
    n_cols = max((len(row) for row in expanded), default=0)
    preview = [row[:max_cols] for row in expanded[:max_rows]]
    return preview, n_rows, n_cols, total_cells


def compare_grids(pred: list[list[str]], gt: list[list[str]]) -> tuple[int, int]:
    rows = max(len(pred), len(gt))
    cols = max(
        max((len(row) for row in pred), default=0),
        max((len(row) for row in gt), default=0),
    )
    compared = 0
    mismatches = 0
    for r in range(rows):
        for c in range(cols):
            pred_cell = pred[r][c] if r < len(pred) and c < len(pred[r]) else ""
            gt_cell = gt[r][c] if r < len(gt) and c < len(gt[r]) else ""
            if not pred_cell and not gt_cell:
                continue
            compared += 1
            if pred_cell != gt_cell:
                mismatches += 1
    return compared, mismatches


def draw_grid(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    title: str,
    grid: list[list[str]],
    other: list[list[str]],
    font,
    font_bold,
    col_w: int,
    row_h: int,
    max_cols: int,
) -> int:
    draw.text((x, y), title, font=font_bold, fill=(20, 20, 20))
    y += 28
    for r, row in enumerate(grid):
        for c in range(max_cols):
            value = row[c] if c < len(row) else ""
            other_value = other[r][c] if r < len(other) and c < len(other[r]) else ""
            fill = (238, 255, 238) if value == other_value else (255, 232, 232)
            outline = (120, 180, 120) if value == other_value else (210, 90, 90)
            left = x + c * col_w
            top = y + r * row_h
            draw.rectangle([left, top, left + col_w, top + row_h], fill=fill, outline=outline, width=1)
            max_len = max(8, int(col_w / 7.0))
            clipped = value[:max_len] + ("..." if len(value) > max_len else "")
            draw.text((left + 4, top + 5), clipped, font=font, fill=(20, 20, 20))
    return y + max(1, len(grid)) * row_h


def draw_boxes(draw: ImageDraw.ImageDraw, rec_items: list[dict[str, Any]], scale: float) -> None:
    for item in rec_items:
        bbox = item.get("bbox")
        if bbox is None:
            bbox = []
        if len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [float(v) * scale for v in bbox]
        draw.rectangle([x0, y0, x1, y1], outline=(210, 0, 180), width=max(1, int(2 * scale)))


def make_example_png(
    *,
    run_dir: Path,
    out_path: Path,
    filename: str,
    teds_s: float,
    teds: float,
    group: str,
    aux_rec: dict[str, list[dict[str, Any]]],
    inference: dict[str, Any],
    box_legend: str | None = None,
) -> None:
    image_path = run_dir / "images" / filename
    image = Image.open(image_path).convert("RGB")
    max_width = 1150
    scale = min(1.0, max_width / image.width)
    display = image.resize((int(image.width * scale), int(image.height * scale)))

    pred_html = inference[filename]["pred_string"]
    gt_html = inference[filename]["answer_string"]
    pred_grid, pred_rows, pred_cols, pred_cells = html_to_grid(pred_html)
    gt_grid, gt_rows, gt_cols, gt_cells = html_to_grid(gt_html)
    compared_cells, mismatched_cells = compare_grids(pred_grid, gt_grid)

    max_cols = max(
        min(18, max((len(row) for row in pred_grid), default=0)),
        min(18, max((len(row) for row in gt_grid), default=0)),
        1,
    )
    row_h = 34
    col_w = 150 if max_cols <= 8 else (92 if max_cols <= 18 else 76)
    grid_rows = max(len(pred_grid), len(gt_grid), 1)
    grid_h = 28 + grid_rows * row_h
    panel_h = 190 + grid_h * 2
    canvas_w = max(display.width, 20 + max_cols * col_w)
    canvas = Image.new("RGB", (canvas_w, display.height + panel_h), "white")
    canvas.paste(display, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw_boxes(draw, aux_rec.get(filename, []), scale)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    y = display.height + 14
    title = f"{group} | TEDS-S={safe_score(teds_s)}, TEDS={safe_score(teds)}"
    draw.text((10, y), title, font=font_bold, fill=(0, 0, 0))
    y += 28
    draw.text((10, y), filename, font=font, fill=(20, 20, 20))
    y += 22
    score_text = (
        f"Rows pred/GT: {pred_rows}/{gt_rows} | Cols pred/GT: {pred_cols}/{gt_cols} | "
        f"Cells pred/GT: {pred_cells}/{gt_cells} | Preview mismatches: {mismatched_cells}/{compared_cells}"
    )
    draw.text((10, y), score_text, font=font, fill=(30, 30, 30))
    y += 24
    shown_text = (
        f"Preview shows rows pred/GT: {len(pred_grid)}/{len(gt_grid)} of {pred_rows}/{gt_rows}; "
        f"columns shown: {max_cols} of pred/GT {pred_cols}/{gt_cols}"
    )
    draw.text((10, y), shown_text, font=font, fill=(80, 80, 80))
    y += 22
    legend = box_legend or "Magenta = PSENet+MASTER OCR text-region boxes"
    draw.text(
        (10, y),
        f"{legend}, count={len(aux_rec.get(filename, []))}",
        font=font,
        fill=(120, 0, 120),
    )
    y += 28
    draw.text(
        (10, y),
        "Red cells differ in this compact preview. Green cells match exactly after plain text extraction.",
        font=font,
        fill=(110, 30, 30),
    )
    y += 34
    max_grid_w = max_cols * col_w
    if max_grid_w > canvas.width - 20:
        col_w = max(90, (canvas.width - 20) // max_cols)
    y = draw_grid(
        draw,
        x=10,
        y=y,
        title="Prediction preview",
        grid=pred_grid,
        other=gt_grid,
        font=font,
        font_bold=font_bold,
        col_w=col_w,
        row_h=row_h,
        max_cols=max_cols,
    )
    y += 28
    y = draw_grid(
        draw,
        x=10,
        y=y,
        title="Ground truth preview",
        grid=gt_grid,
        other=pred_grid,
        font=font,
        font_bold=font_bold,
        col_w=col_w,
        row_h=row_h,
        max_cols=max_cols,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def pick_examples(rows: list[dict[str, Any]], n: int) -> dict[str, list[dict[str, Any]]]:
    by_teds = sorted(rows, key=lambda row: row["teds"], reverse=True)
    by_low = sorted(rows, key=lambda row: row["teds"])
    by_gap = sorted(rows, key=lambda row: row["teds_s"] - row["teds"], reverse=True)
    by_mid = sorted(rows, key=lambda row: abs(row["teds"] - 0.5))
    return {
        "best_teds": by_teds[:n],
        "worst_teds": by_low[:n],
        "structure_ok_text_gap": by_gap[:n],
        "middle_teds": by_mid[:n],
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = args.output_dir or (run_dir / "report")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(run_dir / "ted_score_output.summary.json")
    validation = load_json(run_dir / "post_ocr_validation.json")
    ted_rows_raw = load_json(run_dir / "ted_score_output.json")
    inference = load_json(run_dir / "full_model_inference.json")
    manifest = load_json(run_dir / "manifest.json")
    skipped_path = run_dir / "skipped_samples.json"
    skipped = load_json(skipped_path) if skipped_path.is_file() else []
    with (run_dir / "aux_rec.pkl").open("rb") as f:
        aux_rec = pickle.load(f)

    rows = [
        {
            "filename": row[0],
            "teds_s": float(row[-2]),
            "teds": float(row[-1]),
            "ocr_regions": len(aux_rec.get(row[0], [])),
        }
        for row in ted_rows_raw
    ]

    groups = pick_examples(rows, args.examples_per_group)
    example_manifest = {}
    for group, items in groups.items():
        example_manifest[group] = []
        for idx, item in enumerate(items):
            filename = item["filename"]
            out_path = out_dir / "examples" / group / f"{idx:02d}_{filename}"
            make_example_png(
                run_dir=run_dir,
                out_path=out_path,
                filename=filename,
                teds_s=item["teds_s"],
                teds=item["teds"],
                group=group,
                aux_rec=aux_rec,
                inference=inference,
                box_legend=args.box_legend,
            )
            example_manifest[group].append({**item, "visualization": str(out_path)})

    row_by_name = {row["filename"]: row for row in rows}
    manifest_by_name = {row["filename"]: row for row in manifest}
    merged_manifest = []
    for name, row in sorted(row_by_name.items()):
        merged_manifest.append({**manifest_by_name.get(name, {}), **row})

    (out_dir / "per_table_manifest.json").write_text(
        json.dumps(merged_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "example_manifest.json").write_text(
        json.dumps(example_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    markdown = [
        "# Paper Collection OCR-Style TFLOP Report",
        "",
        f"Pipeline: {args.pipeline_label}",
        "",
        "## Results",
        "",
        f"- Valid evaluated samples: `{summary['num_samples']}`",
        f"- TEDS-S: `{summary['teds_s']:.4f}` ({summary['teds_s'] * 100:.2f})",
        f"- TEDS: `{summary['teds']:.4f}` ({summary['teds'] * 100:.2f})",
        f"- TEDS-S = 1: `{summary['teds_s_1']}`",
        f"- TEDS = 1: `{summary['teds_1']}`",
        "",
        "## Input Validation",
        "",
        f"- Images present: `{validation['images_present']}/{validation['samples']}`",
        f"- OCR text regions: `{validation['total_regions']}`",
        f"- Empty OCR samples: `{validation['empty_rec_samples']}`",
        f"- Invalid bbox regions: `{validation['invalid_bbox_regions']}`",
        f"- Out-of-bounds regions: `{validation['out_of_bounds_regions']}`",
        f"- Skipped before run because GT HTML had no usable table: `{len(skipped)}`",
        "",
        "## Examples",
        "",
        "Example images are grouped under `examples/`:",
        "",
        "- `best_teds`: highest full TEDS",
        "- `worst_teds`: lowest full TEDS",
        "- `structure_ok_text_gap`: high TEDS-S minus TEDS gap, usually structure is better than OCR text/content",
        "- `middle_teds`: examples around TEDS 0.5",
        "",
        "Each example shows the table crop with magenta OCR boxes and short prediction/GT text previews.",
        "",
        "## Notes",
        "",
        "This is an OCR-style run only. The FTN-like annotation-input comparison is not run because this paper collection does not currently provide explicit cell bounding boxes.",
    ]
    (out_dir / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"report_dir": str(out_dir), "samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
