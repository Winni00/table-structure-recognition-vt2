#!/usr/bin/env python3
"""Create side-by-side visuals for MASTER vs PARSeq Paper Collection runs."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
sys.path.insert(0, str(ROOT / "scripts"))
from report_paper_collection_ocr_style_results import make_example_png  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-run", type=Path, required=True)
    parser.add_argument("--parseq-run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-group", type=int, default=6)
    return parser.parse_args()


def load_run_payload(run_dir: Path):
    inference = json.loads((run_dir / "full_model_inference.json").read_text(encoding="utf-8"))
    with (run_dir / "aux_rec.pkl").open("rb") as f:
        aux_rec = pickle.load(f)
    return inference, aux_rec


def header(text: str, width: int, height: int = 96) -> Image.Image:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font_bold = ImageFont.load_default()
        font = font_bold
    y = 10
    for idx, line in enumerate(text.splitlines()):
        draw.text((14, y), line, font=font_bold if idx == 0 else font, fill=(0, 0, 0))
        y += 30 if idx == 0 else 22
    return img


def side_by_side(left: Path, right: Path, out_path: Path, title: str) -> None:
    left_img = Image.open(left).convert("RGB")
    right_img = Image.open(right).convert("RGB")
    max_panel_w = 1250
    panels = []
    for img in [left_img, right_img]:
        scale = min(1.0, max_panel_w / img.width)
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        panels.append(img)
    left_img, right_img = panels
    gap = 24
    width = left_img.width + right_img.width + gap
    height = max(left_img.height, right_img.height)
    head = header(title, width)
    canvas = Image.new("RGB", (width, head.height + height), "white")
    canvas.paste(head, (0, 0))
    canvas.paste(left_img, (0, head.height))
    canvas.paste(right_img, (left_img.width + gap, head.height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def render_single(*, run_dir: Path, out_path: Path, filename: str, teds_s: float, teds: float, group: str, aux_rec, inference, legend: str) -> None:
    make_example_png(
        run_dir=run_dir,
        out_path=out_path,
        filename=filename,
        teds_s=teds_s,
        teds=teds,
        group=group,
        aux_rec=aux_rec,
        inference=inference,
        box_legend=legend,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    master_inf, master_rec = load_run_payload(args.master_run)
    parseq_inf, parseq_rec = load_run_payload(args.parseq_run)

    groups = {
        "parseq_better": analysis["top_parseq_improvements"][: args.examples_per_group],
        "parseq_worse": analysis["top_parseq_drops"][: args.examples_per_group],
        "high_structure_parseq_better": analysis["high_structure_top_parseq_improvements"][: args.examples_per_group],
        "high_structure_parseq_worse": analysis["high_structure_top_parseq_drops"][: args.examples_per_group],
    }

    manifest = {}
    for group, rows in groups.items():
        manifest[group] = []
        for idx, row in enumerate(rows):
            filename = row["filename"]
            stem = f"{idx:02d}_{filename}"
            master_panel = args.output_dir / "single_panels" / group / f"{stem}.master.png"
            parseq_panel = args.output_dir / "single_panels" / group / f"{stem}.parseq.png"
            comparison = args.output_dir / "comparisons" / group / f"{stem}.comparison.png"
            render_single(
                run_dir=args.master_run,
                out_path=master_panel,
                filename=filename,
                teds_s=row["master_teds_s"],
                teds=row["master_teds"],
                group="MASTER OCR text",
                aux_rec=master_rec,
                inference=master_inf,
                legend="Magenta = same PSENet text-region boxes; text source = MASTER OCR",
            )
            render_single(
                run_dir=args.parseq_run,
                out_path=parseq_panel,
                filename=filename,
                teds_s=row["parseq_teds_s"],
                teds=row["parseq_teds"],
                group="PARSeq OCR text",
                aux_rec=parseq_rec,
                inference=parseq_inf,
                legend="Magenta = same PSENet text-region boxes; text source = PARSeq recognizer",
            )
            title = (
                f"{group}: {filename}\n"
                f"MASTER TEDS-S/TEDS={row['master_teds_s']:.4f}/{row['master_teds']:.4f} | "
                f"PARSeq={row['parseq_teds_s']:.4f}/{row['parseq_teds']:.4f} | "
                f"delta={row['delta_teds_s']:+.4f}/{row['delta_teds']:+.4f}"
            )
            side_by_side(master_panel, parseq_panel, comparison, title)
            manifest[group].append({**row, "master_panel": str(master_panel), "parseq_panel": str(parseq_panel), "comparison": str(comparison)})

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# MASTER vs PARSeq Visual Comparisons",
        "",
        "Both variants use the same updated target-domain table crops, best rotation, PSENet boxes, public TFLOP checkpoint, and evaluator. Only the recognized text differs.",
        "",
    ]
    for group, rows in manifest.items():
        lines.append(f"## {group}")
        lines.append("")
        lines.append("| File | MASTER TEDS | PARSeq TEDS | Delta TEDS | Visual |")
        lines.append("|---|---:|---:|---:|---|")
        for row in rows:
            lines.append(f"| `{row['filename']}` | {row['master_teds']:.4f} | {row['parseq_teds']:.4f} | {row['delta_teds']:+.4f} | `{row['comparison']}` |")
        lines.append("")
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "groups": {k: len(v) for k, v in manifest.items()}}, indent=2))


if __name__ == "__main__":
    main()
