#!/usr/bin/env python3
"""Create side-by-side Paper Collection visuals: MASTER vs PyMuPDF text.

Both runs use the same table images, PSENet boxes, public TFLOP checkpoint and
official evaluator. The intended comparison is the text source inside
``aux_rec.pkl``:

- baseline: MASTER OCR text
- experiment: PyMuPDF text extracted from the source PDF at the same boxes
"""

from __future__ import annotations

import json
import pickle
import sys
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
MASTER_RUN = ROOT / "results/tflop_paper_collection_updated_crops_ocr_style_public"
PYMUPDF_RUN = ROOT / "results/tflop_paper_collection_updated_crops_pymupdf_text_public"
OUT_DIR = ROOT / "results/paper_collection_updated_crops_master_vs_pymupdf_visuals"

sys.path.insert(0, str(ROOT / "scripts"))
from report_paper_collection_ocr_style_results import make_example_png  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-run", type=Path, default=MASTER_RUN)
    parser.add_argument("--pymupdf-run", type=Path, default=PYMUPDF_RUN)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def load_scores(run_dir: Path) -> dict[str, dict[str, float]]:
    rows = json.loads((run_dir / "ted_score_output.json").read_text(encoding="utf-8"))
    return {
        row[0]: {
            "teds_s": float(row[-2]),
            "teds": float(row[-1]),
        }
        for row in rows
    }


def load_run_payload(run_dir: Path):
    inference = json.loads((run_dir / "full_model_inference.json").read_text(encoding="utf-8"))
    with (run_dir / "aux_rec.pkl").open("rb") as f:
        aux_rec = pickle.load(f)
    return inference, aux_rec


def header(text: str, width: int, height: int = 92) -> Image.Image:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font_bold = ImageFont.load_default()
        font = font_bold
    lines = text.split("\n")
    y = 12
    for idx, line in enumerate(lines):
        draw.text((14, y), line, font=font_bold if idx == 0 else font, fill=(0, 0, 0))
        y += 30 if idx == 0 else 22
    return img


def side_by_side(left: Path, right: Path, out_path: Path, title: str) -> None:
    left_img = Image.open(left).convert("RGB")
    right_img = Image.open(right).convert("RGB")

    # Keep slide/browser viewing practical while preserving enough detail.
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
    y0 = head.height
    canvas.paste(left_img, (0, y0))
    canvas.paste(right_img, (left_img.width + gap, y0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def render_single(
    *,
    run_dir: Path,
    out_path: Path,
    filename: str,
    scores: dict[str, float],
    inference,
    aux_rec,
    group: str,
    box_legend: str,
) -> None:
    make_example_png(
        run_dir=run_dir,
        out_path=out_path,
        filename=filename,
        teds_s=scores["teds_s"],
        teds=scores["teds"],
        group=group,
        aux_rec=aux_rec,
        inference=inference,
        box_legend=box_legend,
    )


def pick_examples(master_scores, pymupdf_scores):
    common = sorted(set(master_scores) & set(pymupdf_scores))
    rows = []
    for name in common:
        m = master_scores[name]
        p = pymupdf_scores[name]
        rows.append(
            {
                "filename": name,
                "master_teds_s": m["teds_s"],
                "master_teds": m["teds"],
                "pymupdf_teds_s": p["teds_s"],
                "pymupdf_teds": p["teds"],
                "delta_teds_s": p["teds_s"] - m["teds_s"],
                "delta_teds": p["teds"] - m["teds"],
            }
        )

    worse = sorted(rows, key=lambda r: r["delta_teds"])[:6]
    better = sorted(rows, key=lambda r: r["delta_teds"], reverse=True)[:6]
    structure_worse = sorted(rows, key=lambda r: r["delta_teds_s"])[:4]
    return {
        "pymupdf_worse_teds": worse,
        "pymupdf_better_teds": better,
        "pymupdf_worse_teds_s": structure_worse,
    }


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    master_run = args.master_run
    pymupdf_run = args.pymupdf_run
    out_dir.mkdir(parents=True, exist_ok=True)
    master_scores = load_scores(master_run)
    pymupdf_scores = load_scores(pymupdf_run)
    master_inf, master_rec = load_run_payload(master_run)
    pymupdf_inf, pymupdf_rec = load_run_payload(pymupdf_run)

    groups = pick_examples(master_scores, pymupdf_scores)
    manifest = {}
    single_dir = out_dir / "single_panels"
    compare_dir = out_dir / "comparisons"

    for group, items in groups.items():
        manifest[group] = []
        for idx, row in enumerate(items):
            filename = row["filename"]
            stem = f"{idx:02d}_{filename}"
            master_panel = single_dir / group / f"{stem}.master.png"
            pymu_panel = single_dir / group / f"{stem}.pymupdf.png"
            compare_panel = compare_dir / group / f"{stem}.comparison.png"
            render_single(
                run_dir=master_run,
                out_path=master_panel,
                filename=filename,
                scores={"teds_s": row["master_teds_s"], "teds": row["master_teds"]},
                inference=master_inf,
                aux_rec=master_rec,
                group="MASTER OCR text",
                box_legend="Magenta = PSENet text-region boxes; text source = MASTER OCR",
            )
            render_single(
                run_dir=pymupdf_run,
                out_path=pymu_panel,
                filename=filename,
                scores={"teds_s": row["pymupdf_teds_s"], "teds": row["pymupdf_teds"]},
                inference=pymupdf_inf,
                aux_rec=pymupdf_rec,
                group="PyMuPDF PDF text",
                box_legend="Magenta = same PSENet text-region boxes; text source = PyMuPDF PDF extraction",
            )
            title = (
                f"{group}: {filename}\n"
                f"MASTER TEDS-S/TEDS={row['master_teds_s']:.4f}/{row['master_teds']:.4f} | "
                f"PyMuPDF={row['pymupdf_teds_s']:.4f}/{row['pymupdf_teds']:.4f} | "
                f"delta={row['delta_teds_s']:+.4f}/{row['delta_teds']:+.4f}"
            )
            side_by_side(master_panel, pymu_panel, compare_panel, title)
            manifest[group].append(
                {
                    **row,
                    "master_panel": str(master_panel),
                    "pymupdf_panel": str(pymu_panel),
                    "comparison": str(compare_panel),
                }
            )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# MASTER vs PyMuPDF Text Comparison Visuals",
        "",
        "Both variants use the same updated table crops, PSENet text-region boxes, public TFLOP checkpoint, and official TEDS evaluation.",
        "",
        "Only the text source differs:",
        "",
        "- MASTER OCR text",
        "- PyMuPDF text extracted directly from the source PDF at the mapped PSENet boxes",
        "",
        "## Groups",
        "",
    ]
    for group, items in manifest.items():
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| File | MASTER TEDS-S | MASTER TEDS | PyMuPDF TEDS-S | PyMuPDF TEDS | Delta TEDS | Visual |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in items:
            lines.append(
                f"| `{row['filename']}` | {row['master_teds_s']:.4f} | {row['master_teds']:.4f} | "
                f"{row['pymupdf_teds_s']:.4f} | {row['pymupdf_teds']:.4f} | "
                f"{row['delta_teds']:+.4f} | `{row['comparison']}` |"
            )
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "groups": {k: len(v) for k, v in manifest.items()}}, indent=2))


if __name__ == "__main__":
    main()
