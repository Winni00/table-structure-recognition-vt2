#!/usr/bin/env python3
"""Render near-perfect TEDS examples with box alignment and HTML before/after."""

from __future__ import annotations

import html as html_escape
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
TFLOP_REPO = ROOT / "repo" / "TFLOP"
sys.path.insert(0, str(TFLOP_REPO))

from evaluate_ted import strip_html_contents  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from rescore_tflop_fintabnet_canonicalized import flatten_table_sections  # noqa: E402


OUT_DIR = ROOT / "results" / "near_one_alignment_html_debug"

PTN_RUN = ROOT / "results" / "tflop_pubtabnet_val_annotations_filtered_8958"
PTN_IMAGES = ROOT / "data" / "TFLOP-dataset" / "images" / "validation"

FTN_RUN = ROOT / "results" / "tflop_fintabnet_full_annotation_excl_problem_pages"
FTN_IMAGES = FTN_RUN / "images"


def wrap_html(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("<html"):
        return s
    if s.startswith("<table") and s.endswith("</table>"):
        return f"<html><body>{s}</body></html>"
    return f"<html><body><table>{s}</table></body></html>"


def ptn_canonicalize(s: str) -> str:
    return strip_html_contents(wrap_html(s))


def ftn_canonicalize(s: str) -> str:
    return flatten_table_sections(s)


def clean_text(text: str, limit: int = 30) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = " ".join(text.split())
    text = text.encode("ascii", "replace").decode("ascii")
    return text[:limit] + ("..." if len(text) > limit else "")


def select_near_one(ted_path: Path, n: int = 5) -> list[dict[str, Any]]:
    rows = json.loads(ted_path.read_text(encoding="utf-8"))
    candidates = [
        {
            "filename": r[0],
            "edit_distance": r[3],
            "teds_s": float(r[4]),
            "teds": float(r[5]),
        }
        for r in rows
        if float(r[5]) < 1.0
    ]
    candidates.sort(key=lambda x: (-x["teds"], -x["teds_s"], x["filename"]))
    return candidates[:n]


def load_inference(run_dir: Path) -> dict[str, dict[str, str]]:
    return json.loads((run_dir / "full_model_inference.json").read_text(encoding="utf-8"))


def load_rec(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    with (run_dir / "aux_rec.pkl").open("rb") as f:
        return pickle.load(f)


def draw_alignment(
    image_path: Path,
    rec_items: list[dict[str, Any]],
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(2.2, max(1.0, 1000 / max(w, 1)))
    im = img.resize((int(w * scale), int(h * scale)))
    top = 118
    canvas = Image.new("RGB", (im.width, im.height + top + 32), "white")
    canvas.paste(im, (0, top))
    d = ImageDraw.Draw(canvas)

    d.text((10, 8), title, fill=(0, 0, 0))
    d.text((10, 30), subtitle, fill=(75, 75, 75))
    d.text((10, 52), "Origin: (0,0) at top-left. x -> right, y -> down. BBox format: [left, top, right, bottom].", fill=(75, 75, 75))
    d.text((10, 74), f"Table box/crop boundary: [0, 0, {w}, {h}]", fill=(35, 95, 210))
    d.text((10, 96), f"Text-region boxes: {len(rec_items)}", fill=(20, 140, 70))

    # Table/crop boundary
    d.rectangle([0, top, im.width - 1, top + im.height - 1], outline=(35, 95, 210), width=4)

    # Origin marker and axes
    ox, oy = 0, top
    d.ellipse([ox, oy, ox + 10, oy + 10], fill=(225, 30, 30))
    d.line([ox + 5, oy + 5, min(im.width - 5, ox + 105), oy + 5], fill=(225, 30, 30), width=3)
    d.line([ox + 5, oy + 5, ox + 5, min(top + im.height - 5, oy + 85)], fill=(225, 30, 30), width=3)
    d.text((ox + 12, oy + 8), "(0,0)", fill=(225, 30, 30))
    d.text((min(im.width - 40, ox + 108), oy), "x", fill=(225, 30, 30))
    d.text((ox + 10, min(top + im.height - 20, oy + 88)), "y", fill=(225, 30, 30))

    for item in rec_items:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [float(v) * scale for v in bbox]
        y0 += top
        y1 += top
        d.rectangle([x0, y0, x1, y1], outline=(20, 150, 70), width=2)
        label = clean_text(item.get("text", ""))
        if label and (x1 - x0 > 25) and (y1 - y0 > 8):
            d.rectangle([x0, max(top, y0 - 12), min(im.width, x0 + len(label) * 6 + 8), max(top, y0 - 12) + 12], fill=(255, 255, 255))
            d.text((x0 + 2, max(top, y0 - 12)), label, fill=(20, 120, 60))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def html_page(title: str, sections: dict[str, str], meta: dict[str, Any]) -> str:
    def esc(s: str) -> str:
        return html_escape.escape(s or "")

    meta_html = "\n".join(f"<li><b>{esc(str(k))}</b>: {esc(str(v))}</li>" for k, v in meta.items())
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:24px} pre{white-space:pre-wrap;border:1px solid #ccc;padding:12px;max-height:420px;overflow:auto} h2{margin-top:28px}</style>",
        f"<title>{esc(title)}</title></head><body>",
        f"<h1>{esc(title)}</h1><ul>{meta_html}</ul>",
    ]
    for name, value in sections.items():
        parts.append(f"<h2>{esc(name)}</h2><pre>{esc(value)}</pre>")
    parts.append("</body></html>")
    return "\n".join(parts)


def render_dataset(
    name: str,
    run_dir: Path,
    images_dir: Path,
    ted_file: str,
    canonicalize,
    out_subdir: str,
) -> list[dict[str, Any]]:
    selected = select_near_one(run_dir / ted_file, 5)
    inference = load_inference(run_dir)
    rec = load_rec(run_dir)
    out = OUT_DIR / out_subdir
    results = []
    for rank, item in enumerate(selected, 1):
        fn = item["filename"]
        raw_pred = inference[fn]["pred_string"]
        raw_gt = inference[fn]["answer_string"]
        can_pred = canonicalize(raw_pred)
        can_gt = canonicalize(raw_gt)
        stem = f"{rank:02d}_{Path(fn).stem}"
        png_path = out / f"{stem}_boxes_origin.png"
        html_path = out / f"{stem}_html_raw_vs_canonical.html"
        draw_alignment(
            images_dir / fn,
            rec.get(fn, []),
            f"{name} near-one TEDS example {rank}: {fn}",
            f"TEDS-S={item['teds_s']:.6f}, TEDS={item['teds']:.6f}. Blue=table/crop boundary, green=text-region boxes, red=origin/axes.",
            png_path,
        )
        html_path.write_text(
            html_page(
                f"{name} HTML debug: {fn}",
                {
                    "Raw prediction HTML": raw_pred,
                    "Canonicalized prediction HTML": can_pred,
                    "Raw GT HTML": raw_gt,
                    "Canonicalized GT HTML": can_gt,
                },
                {
                    "filename": fn,
                    "TEDS-S": f"{item['teds_s']:.6f}",
                    "TEDS": f"{item['teds']:.6f}",
                    "canonicalization": "TFLOP strip_html_contents" if name.startswith("PTN") else "FTN flatten thead/tbody/tfoot",
                    "box origin": "top-left of image/crop",
                    "table box": f"[0, 0, {Image.open(images_dir / fn).size[0]}, {Image.open(images_dir / fn).size[1]}]",
                    "text-region boxes": len(rec.get(fn, [])),
                },
            ),
            encoding="utf-8",
        )
        results.append({**item, "png": str(png_path), "html": str(html_path), "text_region_boxes": len(rec.get(fn, []))})
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "ptn_val": render_dataset(
            "PTN-Val annotation-input",
            PTN_RUN,
            PTN_IMAGES,
            "ted_score_output.json",
            ptn_canonicalize,
            "ptn_val",
        ),
        "ftn_canonicalized": render_dataset(
            "FTN annotation-input canonicalized",
            FTN_RUN,
            FTN_IMAGES,
            "ted_score_output_ftn_canonicalized.json",
            ftn_canonicalize,
            "ftn_canonicalized",
        ),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Near-One Alignment and HTML Debug", ""]
    for section, items in report.items():
        lines.append(f"## {section}")
        for item in items:
            lines.append(f"- {item['filename']}: TEDS-S={item['teds_s']:.6f}, TEDS={item['teds']:.6f}")
            lines.append(f"  - PNG: `{item['png']}`")
            lines.append(f"  - HTML: `{item['html']}`")
        lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR), "readme": str(OUT_DIR / "README.md")}, indent=2))


if __name__ == "__main__":
    main()
