"""Render FTN OCR-style failure/debug cases after OCR merge."""

from __future__ import annotations

import argparse
import html
import json
import pickle
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RUN_DIR = ROOT / "results/tflop_fintabnet_ocr_style_full_10622"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=80)
    return parser.parse_args()


def normalize_bbox(bbox: Any) -> list[float] | None:
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


def html_page(title: str, blocks: dict[str, str], meta: dict[str, Any]) -> str:
    rows = "\n".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in meta.items()
    )
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.35}table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #aaa;padding:5px 8px;vertical-align:top}pre{white-space:pre-wrap;background:#f6f6f6;border:1px solid #ccc;padding:10px;font-size:12px;max-height:640px;overflow:auto}.warn{color:#9a3412;font-weight:bold}</style>",
        "</head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<table>{rows}</table>",
    ]
    for name, value in blocks.items():
        parts.append(f"<h2>{html.escape(name)}</h2><pre>{html.escape(value)}</pre>")
    parts.append("</body></html>")
    return "\n".join(parts)


def draw_debug_image(
    image_path: Path,
    items: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        small = ImageFont.truetype("DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    draw.rectangle([0, 0, width - 1, height - 1], outline=(30, 90, 220), width=4)
    axis_len = min(70, max(25, width // 8))
    draw.line([0, 0, axis_len, 0], fill=(220, 0, 0), width=3)
    draw.line([0, 0, 0, axis_len], fill=(220, 0, 0), width=3)
    draw.text((4, 4), "(0,0)", fill=(220, 0, 0), font=small)
    draw.text((axis_len + 3, 0), "x", fill=(220, 0, 0), font=small)
    draw.text((4, axis_len + 3), "y", fill=(220, 0, 0), font=small)

    invalid = 0
    out_of_bounds = 0
    for idx, item in enumerate(items):
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None:
            invalid += 1
            continue
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            invalid += 1
            continue
        if x0 < -1 or y0 < -1 or x1 > width + 1 or y1 > height + 1:
            out_of_bounds += 1
        color = (0, 150, 80) if idx < 250 else (240, 130, 0)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        text = str(item.get("text", "")).strip()
        if text:
            label = text[:34]
            lx, ly = x0 + 1, max(0, y0 - 12)
            draw.text((lx, ly), label, fill=color, font=small)

    header_h = 70
    canvas = Image.new("RGB", (width, height + header_h), "white")
    canvas.paste(image, (0, header_h))
    cdraw = ImageDraw.Draw(canvas)
    cdraw.text((10, 8), title, fill=(0, 0, 0), font=font)
    cdraw.text(
        (10, 35),
        f"image={width}x{height}; OCR regions={len(items)}; invalid={invalid}; out_of_bounds={out_of_bounds}; blue=crop boundary; green=OCR text boxes; red=origin",
        fill=(80, 80, 80),
        font=small,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "image_width": width,
        "image_height": height,
        "ocr_regions": len(items),
        "invalid_bbox_regions": invalid,
        "out_of_bounds_regions": out_of_bounds,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = args.output_dir or (run_dir / "ocr_failure_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    aux_json_path = run_dir / "aux.json"
    aux_rec_path = run_dir / "aux_rec.pkl"
    images_dir = run_dir / "images"
    validation_path = run_dir / "post_ocr_validation.json"

    if not aux_json_path.exists():
        raise FileNotFoundError(aux_json_path)
    if not aux_rec_path.exists():
        raise FileNotFoundError(aux_rec_path)
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    aux = json.loads(aux_json_path.read_text(encoding="utf-8"))
    with aux_rec_path.open("rb") as f:
        aux_rec = pickle.load(f)
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}

    candidates: list[dict[str, Any]] = []
    for filename in sorted(aux):
        items = aux_rec.get(filename, [])
        invalid = 0
        out_of_bounds = 0
        width = height = 0
        image_path = images_dir / filename
        if image_path.exists():
            with Image.open(image_path) as image:
                width, height = image.size
        for item in items:
            bbox = normalize_bbox(item.get("bbox"))
            if bbox is None:
                invalid += 1
                continue
            x0, y0, x1, y1 = bbox
            if x1 <= x0 or y1 <= y0:
                invalid += 1
            if width and height and (x0 < -1 or y0 < -1 or x1 > width + 1 or y1 > height + 1):
                out_of_bounds += 1
        if not items or invalid or out_of_bounds:
            candidates.append(
                {
                    "filename": filename,
                    "ocr_regions": len(items),
                    "invalid_bbox_regions": invalid,
                    "out_of_bounds_regions": out_of_bounds,
                    "priority": (0 if not items else 1, -(invalid + out_of_bounds), filename),
                }
            )

    candidates.sort(key=lambda x: x["priority"])
    selected = candidates[: args.limit]
    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "validation": validation,
        "num_candidates": len(candidates),
        "rendered": [],
    }
    for idx, rec in enumerate(selected):
        filename = rec["filename"]
        stem = f"{idx:03d}_{Path(filename).stem}"
        image_path = images_dir / filename
        png_path = out_dir / f"{stem}_ocr_debug.png"
        html_path = out_dir / f"{stem}_html.html"
        draw_stats = draw_debug_image(
            image_path,
            aux_rec.get(filename, []),
            png_path,
            f"FTN OCR debug: {filename}",
        )
        blocks = {
            "GT HTML": aux[filename].get("html", ""),
            "OCR regions JSON": json.dumps(aux_rec.get(filename, []), indent=2, ensure_ascii=False),
        }
        html_path.write_text(
            html_page(
                f"FTN OCR debug: {filename}",
                blocks,
                {
                    **{k: v for k, v in rec.items() if k != "priority"},
                    **draw_stats,
                    "png": png_path.name,
                },
            ),
            encoding="utf-8",
        )
        manifest["rendered"].append({**rec, **draw_stats, "png": str(png_path), "html": str(html_path)})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = ["# FTN OCR Failure Debug", ""]
    readme.append(f"- Candidates: {len(candidates)}")
    readme.append(f"- Rendered: {len(selected)}")
    readme.append("")
    for item in manifest["rendered"]:
        readme.append(
            f"- {item['filename']}: regions={item['ocr_regions']}, invalid={item['invalid_bbox_regions']}, out_of_bounds={item['out_of_bounds_regions']}"
        )
        readme.append(f"  - PNG: `{item['png']}`")
        readme.append(f"  - HTML: `{item['html']}`")
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": len(candidates), "rendered": len(selected), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
