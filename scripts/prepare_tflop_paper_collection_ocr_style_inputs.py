"""Prepare local paper-collection tables for OCR-style TFLOP inference.

The input collection is expected to contain one directory per paper with
`*_table_*.png` table crops and matching `*_table_*.html` ground truth HTML.
This script creates the TFLOP aux.json/subset/images layout used by the
PTN-test-like OCR path:

table PNG -> PSENet+MASTER aux_rec.pkl -> TFLOP inference -> TEDS.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RAW_ROOT = ROOT / "data/paper_collection_raw_new"
DEFAULT_OUT_DIR = ROOT / "results/tflop_paper_collection_ocr_style_707"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking them into output_dir/images.",
    )
    return parser.parse_args()


def infer_html_type(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for cell in soup.find_all(["td", "th"]):
        for attr in ("rowspan", "colspan"):
            try:
                if int(cell.get(attr, "1")) > 1:
                    return "complex"
            except ValueError:
                return "complex"
    return "simple"


def wrap_html_if_needed(html: str) -> str:
    html = html.lstrip("\ufeff").strip()
    if "<html" in html.lower():
        return html
    return f"<html><body>{html}</body></html>"


def has_table(html: str) -> bool:
    return BeautifulSoup(html, "html.parser").find("table") is not None


def unique_name(table_png: Path) -> str:
    # Keep names filesystem-safe and unique across paper folders.
    paper_id = table_png.parent.name
    return f"{paper_id}__{table_png.name}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = args.output_dir / "images"
    if image_dir.exists() or image_dir.is_symlink():
        if image_dir.is_symlink() or image_dir.is_file():
            image_dir.unlink()
        else:
            shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    pngs = sorted(args.raw_root.glob("*/*_table_*.png"))
    for png in pngs:
        html_path = png.with_suffix(".html")
        xml_path = png.with_suffix(".xml")
        if not html_path.exists():
            continue
        candidates.append((png, html_path, xml_path if xml_path.exists() else None))

    selected = candidates[args.offset :]
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]

    aux: dict[str, dict[str, str]] = {}
    manifest = []
    skipped = []
    for png, html_path, xml_path in selected:
        name = unique_name(png)
        html = wrap_html_if_needed(html_path.read_text(encoding="utf-8"))
        if not has_table(html):
            skipped.append(
                {
                    "filename": name,
                    "reason": "gt_html_has_no_table",
                    "table_png": str(png),
                    "gt_html": str(html_path),
                }
            )
            continue
        aux[name] = {"html": html, "type": infer_html_type(html)}

        target = image_dir / name
        if args.copy_images:
            shutil.copy2(png, target)
            image_mode = "copy"
        else:
            os.symlink(png.resolve(), target)
            image_mode = "symlink"

        manifest.append(
            {
                "filename": name,
                "paper_id": png.parent.name,
                "table_png": str(png),
                "gt_html": str(html_path),
                "gt_xml": str(xml_path) if xml_path else None,
                "image_mode": image_mode,
                "html_type": aux[name]["type"],
            }
        )

    aux_path = args.output_dir / "aux.json"
    aux_path.write_text(json.dumps(aux, ensure_ascii=False), encoding="utf-8")

    subset_path = args.output_dir / "subset.txt"
    subset_path.write_text("\n".join(sorted(aux)) + "\n", encoding="utf-8")

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "skipped_samples.json").write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "raw_root": str(args.raw_root),
        "output_dir": str(args.output_dir),
        "total_table_png": len(pngs),
        "samples_with_png_and_html": len(candidates),
        "selected_samples": len(selected),
        "valid_samples": len(aux),
        "skipped_samples": len(skipped),
        "aux_json": str(aux_path),
        "subset": str(subset_path),
        "images_dir": str(image_dir),
        "skipped_samples_json": str(args.output_dir / "skipped_samples.json"),
        "html_type_counts": {
            "simple": sum(1 for v in aux.values() if v["type"] == "simple"),
            "complex": sum(1 for v in aux.values() if v["type"] == "complex"),
        },
        "purpose": "Paper collection OCR-style TFLOP run with TEDS-capable PNG+GT-HTML subset.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
