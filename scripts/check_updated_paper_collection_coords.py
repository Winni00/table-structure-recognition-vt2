"""Check TargetDomain' updated paper-collection crop coordinates.

The updated collection contains:

- one directory per paper
- table crop PNGs (`*_table_*.png`)
- page context PNGs (`page_<n>_*_table_*.png`)
- matching GT HTML/XML
- `table_coord.json` with PDF-page crop coordinates

This script builds a manifest and renders visual sanity checks. It does not
run TFLOP. The goal is to verify whether crop coordinates, page images and
table crops agree before using the data for OCR-style TFLOP inference.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_DATA_ROOT = ROOT / "data/paper_collection_updated_crops/Data"
DEFAULT_OUT_DIR = ROOT / "results/paper_collection_updated_crops_coord_check"

PROBLEMATIC_DOIS = {
    "10.1016%j.bjid.2012.09.004",
    "10.1016%j.fm.2018.07.004",
    "10.1016%j.foodcont.2017.10.028",
    "10.1016%j.ijfoodmicro.2008.03.029",
    "10.1016%j.ijfoodmicro.2020.108750",
    "10.1016%j.meegid.2014.11.003",
    "10.1016%j.scitotenv.2004.09.025",
    "10.1016%j.vetmic.2011.11.009",
    "10.1016%j.vetmic.2014.02.045",
    "10.1016%j.vetmic.2017.06.010",
    "10.1186%s12866-017-0938-1",
    "10.1186%s12941-017-0242-9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--normal-limit", type=int, default=12)
    parser.add_argument(
        "--problem-limit",
        type=int,
        default=24,
        help="Max number of TargetDomain-listed header-risk samples to visualize.",
    )
    return parser.parse_args()


def pdf_page_size(pdf_path: Path, page_no: int) -> tuple[float, float] | None:
    """Return page size in PDF points for a page using poppler pdfinfo."""
    try:
        proc = subprocess.run(
            ["pdfinfo", "-f", str(page_no), "-l", str(page_no), str(pdf_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    # Example: Page    2 size: 595.276 x 790.866 pts
    match = re.search(
        rf"Page\s+{page_no}\s+size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
        proc.stdout,
    )
    if not match:
        match = re.search(r"Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts", proc.stdout)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def bbox_fit_to_page(
    coord: dict[str, Any],
    page_img_size: tuple[int, int],
    pdf_size: tuple[float, float],
) -> list[float]:
    """Map BOTTOMLEFT PDF coordinates to page-image pixels by page-size ratio."""
    img_w, img_h = page_img_size
    pdf_w, pdf_h = pdf_size
    sx = img_w / pdf_w
    sy = img_h / pdf_h
    return [
        coord["left"] * sx,
        (pdf_h - coord["top"]) * sy,
        coord["right"] * sx,
        (pdf_h - coord["bottom"]) * sy,
    ]


def bbox_raw(coord: dict[str, Any], page_img_h: int, factor: float = 1.0) -> list[float]:
    """Map BOTTOMLEFT coordinates directly as pixels, optionally scaled."""
    return [
        coord["left"] * factor,
        page_img_h - coord["top"] * factor,
        coord["right"] * factor,
        page_img_h - coord["bottom"] * factor,
    ]


def draw_bbox(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    color: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
    width: int = 4,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    draw.text((x0 + 4, max(0, y0 - 14)), label, fill=color, font=font)


def thumb(image: Image.Image, max_w: int = 850) -> Image.Image:
    image = image.convert("RGB")
    if image.width <= max_w:
        return image
    ratio = max_w / image.width
    return image.resize((max_w, int(image.height * ratio)))


def render_visual(sample: dict[str, Any], output_path: Path) -> None:
    font = ImageFont.load_default()
    page_img = Image.open(sample["page_png"]).convert("RGB")
    crop_img = Image.open(sample["table_png"]).convert("RGB")
    draw = ImageDraw.Draw(page_img)

    if sample.get("fit_box_px"):
        draw_bbox(draw, sample["fit_box_px"], (0, 80, 255), "fit PDF coords to page image", font)
    draw_bbox(draw, sample["raw_box_1x_px"], (255, 0, 0), "raw coords x1", font, width=2)
    draw_bbox(draw, sample["raw_box_2x_px"], (255, 140, 0), "raw coords x2", font, width=2)

    page_thumb = thumb(page_img, max_w=950)
    crop_thumb = thumb(crop_img, max_w=950)

    pad = 18
    text_h = 135
    w = max(page_thumb.width, crop_thumb.width) + 2 * pad
    h = text_h + page_thumb.height + crop_thumb.height + 3 * pad
    canvas = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(canvas)
    lines = [
        f"{sample['sample_id']} | header-risk={sample['header_risk']} | page={sample['page_no']}",
        f"coord_origin={sample.get('coord_origin')} | crop={sample['crop_size_px']} | page_img={sample['page_image_size_px']}",
        f"PDF page size={sample.get('pdf_page_size_pts')} | fit/crop ratio={sample.get('fit_to_crop_ratio')}",
        "Blue = PDF-coordinate box fitted to page image. Red/orange = raw coords x1/x2 sanity checks.",
    ]
    y = pad
    for line in lines:
        d.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += 18
    canvas.paste(page_thumb, (pad, text_h))
    canvas.paste(crop_thumb, (pad, text_h + page_thumb.height + pad))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def collect_samples(data_root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for coord_path in sorted(data_root.glob("*/table_coord.json")):
        paper_dir = coord_path.parent
        paper_id = paper_dir.name
        pdfs = sorted(paper_dir.glob("*.pdf"))
        pdf_path = pdfs[0] if pdfs else None
        coords = json.loads(coord_path.read_text(encoding="utf-8"))
        for sample_id, coord in sorted(coords.items()):
            table_png = paper_dir / f"{sample_id}.png"
            html_path = paper_dir / f"{sample_id}.html"
            xml_path = paper_dir / f"{sample_id}.xml"
            page_no = int(coord["page_no"])
            page_png = paper_dir / f"page_{page_no}_{sample_id}.png"
            if not table_png.exists() or not html_path.exists():
                continue
            item = {
                "sample_id": sample_id,
                "paper_id": paper_id,
                "table_png": str(table_png),
                "page_png": str(page_png) if page_png.exists() else None,
                "gt_html": str(html_path),
                "gt_xml": str(xml_path) if xml_path.exists() else None,
                "pdf": str(pdf_path) if pdf_path else None,
                "header_risk": paper_id in PROBLEMATIC_DOIS,
                "coord": coord,
                "coord_origin": coord.get("coord_origin"),
                "page_no": page_no,
            }
            with Image.open(table_png) as im:
                item["crop_size_px"] = list(im.size)
            if page_png.exists():
                with Image.open(page_png) as im:
                    page_size = im.size
                item["page_image_size_px"] = list(page_size)
                item["raw_box_1x_px"] = bbox_raw(coord, page_size[1], 1.0)
                item["raw_box_2x_px"] = bbox_raw(coord, page_size[1], 2.0)
                if pdf_path:
                    pdf_size = pdf_page_size(pdf_path, page_no)
                    if pdf_size:
                        fit_box = bbox_fit_to_page(coord, page_size, pdf_size)
                        item["pdf_page_size_pts"] = [round(pdf_size[0], 3), round(pdf_size[1], 3)]
                        item["fit_box_px"] = [round(v, 2) for v in fit_box]
                        fit_w = max(1.0, fit_box[2] - fit_box[0])
                        fit_h = max(1.0, fit_box[3] - fit_box[1])
                        item["fit_to_crop_ratio"] = [
                            round(item["crop_size_px"][0] / fit_w, 3),
                            round(item["crop_size_px"][1] / fit_h, 3),
                        ]
            samples.append(item)
    return samples


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(args.data_root)
    normal = [s for s in samples if not s["header_risk"] and s.get("page_png")]
    risky = [s for s in samples if s["header_risk"] and s.get("page_png")]
    selected = normal[: args.normal_limit] + risky[: args.problem_limit]

    visuals = []
    for idx, sample in enumerate(selected):
        safe_name = sample["sample_id"].replace("/", "_")
        out = args.output_dir / "visuals" / f"{idx:03d}_{safe_name}.png"
        render_visual(sample, out)
        sample["visual"] = str(out)
        visuals.append(sample)

    (args.output_dir / "manifest.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "visual_manifest.json").write_text(
        json.dumps(visuals, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ratios = [s.get("fit_to_crop_ratio") for s in samples if s.get("fit_to_crop_ratio")]
    ratio_w = [r[0] for r in ratios]
    ratio_h = [r[1] for r in ratios]
    summary = {
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "samples": len(samples),
        "header_risk_samples": sum(1 for s in samples if s["header_risk"]),
        "samples_with_page_png": sum(1 for s in samples if s.get("page_png")),
        "visualized_samples": len(visuals),
        "fit_to_crop_ratio_median": [
            sorted(ratio_w)[len(ratio_w) // 2] if ratio_w else None,
            sorted(ratio_h)[len(ratio_h) // 2] if ratio_h else None,
        ],
        "fit_to_crop_ratio_minmax": [
            [min(ratio_w), max(ratio_w)] if ratio_w else None,
            [min(ratio_h), max(ratio_h)] if ratio_h else None,
        ],
        "problematic_paper_ids_from_target_domain": sorted(PROBLEMATIC_DOIS),
        "manifest": str(args.output_dir / "manifest.json"),
        "visual_manifest": str(args.output_dir / "visual_manifest.json"),
        "visuals_dir": str(args.output_dir / "visuals"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    readme = [
        "# Updated Paper Collection Coordinate Check",
        "",
        "Purpose: check whether TargetDomain' `table_coord.json` crop coordinates align with the new table crops and page context images.",
        "",
        "Color legend in visuals:",
        "- Blue: PDF crop coordinates fitted to the page image by PDF page size",
        "- Red: raw PDF coordinates used as pixels",
        "- Orange: raw PDF coordinates multiplied by 2",
        "",
        "If the blue box covers the same table as the crop preview, the coordinate mapping is consistent. If red/orange is closer, a raw scale assumption may be needed.",
        "",
        "Summary:",
        f"- samples: `{summary['samples']}`",
        f"- header-risk samples: `{summary['header_risk_samples']}`",
        f"- median crop/fit ratio: `{summary['fit_to_crop_ratio_median']}`",
        f"- visuals: `{summary['visuals_dir']}`",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
