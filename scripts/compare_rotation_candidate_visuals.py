"""Create visual comparison panels for rotation-candidate results."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
RUN_DIR = ROOT / "results/tflop_paper_collection_updated_crops_rotation_candidates_public"
OUT_DIR = RUN_DIR / "rotation_visuals"

sys.path.insert(0, str(ROOT / "scripts"))
from report_paper_collection_ocr_style_results import make_example_png  # noqa: E402


def header(text: str, width: int, height: int = 95) -> Image.Image:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 21)
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        bold = ImageFont.load_default()
        font = bold
    y = 10
    for idx, line in enumerate(text.split("\n")):
        draw.text((10, y), line, font=bold if idx == 0 else font, fill=(0, 0, 0))
        y += 30 if idx == 0 else 21
    return img


def combine(panels: list[Path], out: Path, title: str) -> None:
    imgs = [Image.open(path).convert("RGB") for path in panels]
    max_panel_w = 900
    resized = []
    for img in imgs:
        scale = min(1.0, max_panel_w / img.width)
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        resized.append(img)
    gap = 18
    width = sum(img.width for img in resized) + gap * (len(resized) - 1)
    height = max(img.height for img in resized)
    head = header(title, width)
    canvas = Image.new("RGB", (width, head.height + height), "white")
    canvas.paste(head, (0, 0))
    x = 0
    for img in resized:
        canvas.paste(img, (x, head.height))
        x += img.width + gap
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    summary = json.loads((RUN_DIR / "rotation_summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    inference = json.loads((RUN_DIR / "full_model_inference.json").read_text(encoding="utf-8"))
    with (RUN_DIR / "aux_rec.pkl").open("rb") as f:
        aux_rec = pickle.load(f)

    best_rows = sorted(
        rows,
        key=lambda row: row["delta_best_vs_rot0_teds"] or 0.0,
        reverse=True,
    )[:8]
    manifest = []
    for idx, row in enumerate(best_rows):
        panels = []
        for variant in row["variants"]:
            filename = variant["filename"]
            panel = OUT_DIR / "single" / f"{idx:02d}_{filename}.png"
            make_example_png(
                run_dir=RUN_DIR,
                out_path=panel,
                filename=filename,
                teds_s=variant["teds_s"],
                teds=variant["teds"],
                group=f"rotation {variant['rotation']} deg",
                aux_rec=aux_rec,
                inference=inference,
                box_legend="Magenta = PSENet+MASTER boxes/text after this image rotation",
            )
            panels.append(panel)
        title = (
            f"rotation_test: {row['base_filename']}\n"
            f"rot0 TEDS={row['rot0_teds']:.4f} | best rot={row['best_rotation']} "
            f"TEDS={row['best_teds']:.4f} | delta={row['delta_best_vs_rot0_teds']:+.4f}"
        )
        out = OUT_DIR / "comparisons" / f"{idx:02d}_{row['base_filename']}.comparison.png"
        combine(panels, out, title)
        manifest.append({**row, "comparison": str(out)})

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Rotation Candidate Visuals",
        "",
        "Each comparison shows 0°, 90°, and 270° variants for the same table and GT HTML.",
        "",
        "| File | Best rotation | rot0 TEDS | best TEDS | Delta | Visual |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in manifest:
        lines.append(
            f"| `{row['base_filename']}` | {row['best_rotation']} | "
            f"{row['rot0_teds']:.4f} | {row['best_teds']:.4f} | "
            f"{row['delta_best_vs_rot0_teds']:+.4f} | `{row['comparison']}` |"
        )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR), "examples": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
