"""Prepare broad rotation candidates from low-scoring Paper Collection cases.

This is a diagnostic dataset. It selects all samples whose current
GT-canonicalized score is low, then creates 0/90/270 degree image variants
with the same GT HTML. The goal is to identify rotation-related failures
before running further PyMuPDF-vs-MASTER comparisons.
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
DEFAULT_OUT_DIR = ROOT / "results/tflop_paper_collection_updated_crops_rotation_broad_badcases_public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-teds", type=float, default=0.70)
    parser.add_argument("--max-teds-s", type=float, default=0.80)
    parser.add_argument("--num-ocr-shards", type=int, default=6)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_scores(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        row[0]: {
            "teds_s": float(row[-2]),
            "teds": float(row[-1]),
            "raw": row,
        }
        for row in rows
    }


def as_bbox(item: dict[str, Any]) -> list[float] | None:
    bbox = item.get("bbox")
    if bbox is None:
        return None
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    if len(bbox) != 4:
        return None
    return [float(v) for v in bbox]


def box_stats(items: list[dict[str, Any]]) -> dict[str, float]:
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
    return {
        "box_count": float(n_boxes),
        "vertical_box_ratio": vertical / max(1, n_boxes),
        "horizontal_box_ratio": horizontal / max(1, n_boxes),
    }


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


def write_ocr_shards(output_dir: Path, filenames: list[str], num_shards: int) -> None:
    shard_dir = output_dir / "ocr_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    summary = {"samples": len(filenames), "num_shards": num_shards, "shards": []}
    for idx in range(num_shards):
        start = (idx * len(filenames)) // num_shards
        end = ((idx + 1) * len(filenames)) // num_shards
        shard_names = filenames[start:end]
        subset_path = shard_dir / f"subset_{idx:02d}.txt"
        subset_path.write_text("\n".join(shard_names) + "\n", encoding="utf-8")
        summary["shards"].append(
            {"index": idx, "samples": len(shard_names), "subset": str(subset_path)}
        )
    (shard_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output exists, pass --force to overwrite: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    base_aux: dict[str, Any] = json.loads(
        (args.base_run_dir / "aux.json").read_text(encoding="utf-8")
    )
    with (args.base_run_dir / "aux_rec.pkl").open("rb") as f:
        aux_rec = pickle.load(f)
    score_path = args.base_run_dir / "gt_canonicalized_rescore" / "ted_score_output.json"
    scores = load_scores(score_path)

    selected_names = {
        filename
        for filename, score in scores.items()
        if score["teds"] < args.max_teds or score["teds_s"] < args.max_teds_s
    }
    selected_names.update(args.include)
    selected = []
    missing = []
    for filename in sorted(selected_names):
        src = args.base_run_dir / "images" / filename
        if filename not in base_aux or not src.exists():
            missing.append(filename)
            continue
        with Image.open(src) as image:
            width, height = image.size
        stats = box_stats(aux_rec.get(filename, []))
        score = scores.get(filename, {"teds_s": None, "teds": None})
        selected.append(
            {
                "base_filename": filename,
                "width": width,
                "height": height,
                "aspect_ratio_h_over_w": height / max(1, width),
                "baseline_teds_s": score["teds_s"],
                "baseline_teds": score["teds"],
                **stats,
                "selection_reason": (
                    "explicit_include"
                    if filename in args.include
                    else f"teds<{args.max_teds} or teds_s<{args.max_teds_s}"
                ),
            }
        )

    rotations = [0, 90, 270]
    aux_out: dict[str, Any] = {}
    manifest = []
    for row in selected:
        base_filename = row["base_filename"]
        src = args.base_run_dir / "images" / base_filename
        for rotation in rotations:
            out_name = rotation_filename(base_filename, rotation)
            dst = image_dir / out_name
            rotate_image(src, dst, rotation)
            aux_out[out_name] = base_aux[base_filename]
            manifest.append(
                {
                    **row,
                    "filename": out_name,
                    "rotation": rotation,
                    "source_image": str(src),
                    "image": str(dst),
                }
            )

    filenames = sorted(aux_out)
    (args.output_dir / "aux.json").write_text(
        json.dumps(aux_out, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "subset.txt").write_text(
        "\n".join(filenames) + "\n", encoding="utf-8"
    )
    (args.output_dir / "rotation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "selected_base_candidates.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "missing_candidates.json").write_text(
        json.dumps(missing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_ocr_shards(args.output_dir, filenames, args.num_ocr_shards)

    summary = {
        "base_run_dir": str(args.base_run_dir),
        "score_path": str(score_path),
        "output_dir": str(args.output_dir),
        "max_teds": args.max_teds,
        "max_teds_s": args.max_teds_s,
        "selected_base_candidates": len(selected),
        "missing_candidates": len(missing),
        "rotations": rotations,
        "total_variant_samples": len(aux_out),
        "num_ocr_shards": args.num_ocr_shards,
        "purpose": "Broad rotation diagnostic over all low-scoring Paper Collection samples.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
