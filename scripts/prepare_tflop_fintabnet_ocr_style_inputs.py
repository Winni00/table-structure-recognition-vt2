"""Prepare FinTabNet table crops for an OCR-style TFLOP run.

This does not run PSENet, MASTER, or TFLOP inference. It reuses the already
rendered FinTabNet table crops and GT HTML from the annotation-input adapter,
then writes a subset manifest for the PSENet+MASTER OCR step.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_SOURCE_DIR = BASE_DIR / "results/tflop_fintabnet_full_annotation_excl_problem_pages"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results/tflop_fintabnet_ocr_style_smoke100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of samples to select. Use 0 or a negative value for all samples.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy selected crop images instead of symlinking the source image folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_aux_path = args.source_dir / "aux.json"
    source_images_dir = args.source_dir / "images"
    if not source_aux_path.exists():
        raise FileNotFoundError(source_aux_path)
    if not source_images_dir.exists():
        raise FileNotFoundError(source_images_dir)

    source_aux = json.loads(source_aux_path.read_text(encoding="utf-8"))
    filenames = sorted(source_aux)
    selected = filenames[args.offset :]
    if args.limit is not None and args.limit > 0:
        selected = selected[: args.limit]

    aux = {name: source_aux[name] for name in selected}
    aux_path = args.output_dir / "aux.json"
    aux_path.write_text(json.dumps(aux, ensure_ascii=False), encoding="utf-8")

    subset_path = args.output_dir / "subset.txt"
    subset_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

    images_link = args.output_dir / "images"
    if images_link.exists() or images_link.is_symlink():
        if images_link.is_symlink() or images_link.is_file():
            images_link.unlink()
        else:
            shutil.rmtree(images_link)

    if args.copy_images:
        images_link.mkdir(parents=True, exist_ok=True)
        for name in selected:
            shutil.copy2(source_images_dir / name, images_link / name)
        images_mode = "copied_subset"
    else:
        os.symlink(source_images_dir.resolve(), images_link, target_is_directory=True)
        images_mode = "symlink_to_source"

    summary = {
        "samples": len(selected),
        "offset": args.offset,
        "limit": args.limit,
        "source_dir": str(args.source_dir),
        "source_aux": str(source_aux_path),
        "source_images_dir": str(source_images_dir),
        "output_dir": str(args.output_dir),
        "aux_json": str(aux_path),
        "subset": str(subset_path),
        "images": str(images_link),
        "images_mode": images_mode,
        "purpose": (
            "FinTabNet OCR-style setup: keep FTN table crops and GT HTML, "
            "then generate aux_rec.pkl via PSENet+MASTER instead of annotation cell boxes."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
