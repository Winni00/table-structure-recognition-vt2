"""Prepare rotated Paper Collection candidate images for a TFLOP OCR-style test.

This is a targeted diagnostic, not the default dataset preparation. It selects
tables that look rotation-prone from the existing PSENet OCR run, then creates
0/90/270 degree variants with the same GT HTML. The goal is to test whether
rotating vertical/portrait table crops before PSENet+MASTER improves the final
TEDS scores.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_BASE_RUN = ROOT / "results/tflop_paper_collection_updated_crops_ocr_style_public"
DEFAULT_PYMUPDF_RUN = ROOT / "results/tflop_paper_collection_updated_crops_pymupdf_text_public"
DEFAULT_OUT_DIR = ROOT / "results/tflop_paper_collection_updated_crops_rotation_candidates_public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument("--pymupdf-run-dir", type=Path, default=DEFAULT_PYMUPDF_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--aspect-threshold", type=float, default=2.0)
    parser.add_argument("--vertical-ratio-threshold", type=float, default=0.65)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--copy-zero", action="store_true")
    return parser.parse_args()


def as_bbox(item: dict[str, Any]) -> list[float] | None:
    bbox = item.get("bbox")
    if bbox is None:
        return None
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    if len(bbox) != 4:
        return None
    return [float(v) for v in bbox]


def load_teds(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row[0]: {"teds_s": float(row[-2]), "teds": float(row[-1])} for row in rows}


def candidate_stats(base_run: Path, pymupdf_run: Path) -> list[dict[str, Any]]:
    aux = json.loads((base_run / "aux.json").read_text(encoding="utf-8"))
    with (base_run / "aux_rec.pkl").open("rb") as f:
        aux_rec = pickle.load(f)
    master_scores = load_teds(base_run / "ted_score_output.json")
    pymupdf_scores = load_teds(pymupdf_run / "ted_score_output.json")

    rows = []
    for filename in sorted(aux):
        image_path = base_run / "images" / filename
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        items = aux_rec.get(filename, [])
        vertical = 0
        horizontal = 0
        n_boxes = 0
        for item in items:
            bbox = as_bbox(item)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            bw = max(1e-6, x1 - x0)
            bh = max(1e-6, y1 - y0)
            n_boxes += 1
            if bh > 1.5 * bw:
                vertical += 1
            if bw > 1.5 * bh:
                horizontal += 1
        aspect = height / max(1, width)
        vertical_ratio = vertical / max(1, n_boxes)
        horizontal_ratio = horizontal / max(1, n_boxes)
        master = master_scores.get(filename, {})
        pymupdf = pymupdf_scores.get(filename, {})
        rows.append(
            {
                "filename": filename,
                "width": width,
                "height": height,
                "aspect_ratio_h_over_w": aspect,
                "box_count": n_boxes,
                "vertical_box_ratio": vertical_ratio,
                "horizontal_box_ratio": horizontal_ratio,
                "master_teds_s": master.get("teds_s"),
                "master_teds": master.get("teds"),
                "pymupdf_teds_s": pymupdf.get("teds_s"),
                "pymupdf_teds": pymupdf.get("teds"),
                "pymupdf_delta_teds": (
                    pymupdf["teds"] - master["teds"]
                    if "teds" in pymupdf and "teds" in master
                    else None
                ),
            }
        )
    return rows


def rotation_filename(filename: str, rotation: int) -> str:
    path = Path(filename)
    return f"{path.stem}__rot{rotation}{path.suffix}"


def rotate_image(src: Path, dst: Path, rotation: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if rotation == 0:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)
        return
    with Image.open(src) as image:
        if rotation == 90:
            rotated = image.transpose(Image.Transpose.ROTATE_90)
        elif rotation == 270:
            rotated = image.transpose(Image.Transpose.ROTATE_270)
        else:
            raise ValueError(f"Unsupported rotation: {rotation}")
        rotated.save(dst)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    base_aux = json.loads((args.base_run_dir / "aux.json").read_text(encoding="utf-8"))
    rows = candidate_stats(args.base_run_dir, args.pymupdf_run_dir)
    selected = [
        row
        for row in rows
        if row["aspect_ratio_h_over_w"] >= args.aspect_threshold
        or row["vertical_box_ratio"] >= args.vertical_ratio_threshold
    ]
    selected.sort(
        key=lambda row: (
            row["aspect_ratio_h_over_w"] >= args.aspect_threshold,
            row["aspect_ratio_h_over_w"],
            row["vertical_box_ratio"],
        ),
        reverse=True,
    )
    if args.max_candidates > 0:
        selected = selected[: args.max_candidates]

    rotations = [0, 90, 270]
    aux_out: dict[str, Any] = {}
    manifest = []
    for row in selected:
        filename = row["filename"]
        src = args.base_run_dir / "images" / filename
        for rotation in rotations:
            out_name = rotation_filename(filename, rotation)
            dst = image_dir / out_name
            rotate_image(src, dst, rotation)
            aux_out[out_name] = base_aux[filename]
            manifest.append(
                {
                    **row,
                    "base_filename": filename,
                    "filename": out_name,
                    "rotation": rotation,
                    "source_image": str(src),
                    "image": str(dst),
                }
            )

    (args.output_dir / "aux.json").write_text(
        json.dumps(aux_out, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "subset.txt").write_text(
        "\n".join(sorted(aux_out)) + "\n", encoding="utf-8"
    )
    (args.output_dir / "rotation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "candidate_stats.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "base_run_dir": str(args.base_run_dir),
        "pymupdf_run_dir": str(args.pymupdf_run_dir),
        "output_dir": str(args.output_dir),
        "all_samples": len(rows),
        "selected_base_candidates": len(selected),
        "rotations": rotations,
        "total_variant_samples": len(aux_out),
        "aspect_threshold": args.aspect_threshold,
        "vertical_ratio_threshold": args.vertical_ratio_threshold,
        "purpose": "Test whether rotating vertical/portrait Paper Collection table crops improves PSENet+MASTER OCR-style TFLOP TEDS.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
