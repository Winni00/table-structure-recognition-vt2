"""Validate a TFLOP OCR-style aux bundle before running inference."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aux-json", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--aux-rec-pkl", type=Path, default=None)
    parser.add_argument("--require-aux-rec", action="store_true")
    parser.add_argument(
        "--allow-empty-rec",
        action="store_true",
        help="Allow samples with no OCR regions, but still count/report them.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
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


def main() -> None:
    args = parse_args()
    aux = json.loads(args.aux_json.read_text(encoding="utf-8"))
    filenames = sorted(aux)

    errors: list[str] = []
    image_sizes: dict[str, tuple[int, int]] = {}
    for filename in filenames:
        html = aux[filename].get("html", "")
        if not html.strip():
            errors.append(f"{filename}: empty GT html")
        if "<table" not in html.lower():
            errors.append(f"{filename}: GT html has no <table>")
        image_path = args.images_dir / filename
        if not image_path.exists():
            errors.append(f"{filename}: missing image")
            continue
        try:
            with Image.open(image_path) as image:
                image_sizes[filename] = image.size
        except Exception as exc:  # noqa: BLE001 - validator should report all IO failures
            errors.append(f"{filename}: cannot open image: {exc}")

    rec_stats = {
        "aux_rec_present": False,
        "missing_rec_keys": 0,
        "extra_rec_keys": 0,
        "empty_rec_samples": 0,
        "total_regions": 0,
        "empty_text_regions": 0,
        "invalid_bbox_regions": 0,
        "out_of_bounds_regions": 0,
    }
    if args.aux_rec_pkl and args.aux_rec_pkl.exists():
        rec_stats["aux_rec_present"] = True
        with args.aux_rec_pkl.open("rb") as f:
            aux_rec = pickle.load(f)
        missing = sorted(set(filenames) - set(aux_rec))
        extra = sorted(set(aux_rec) - set(filenames))
        rec_stats["missing_rec_keys"] = len(missing)
        rec_stats["extra_rec_keys"] = len(extra)
        for filename in missing[:20]:
            errors.append(f"{filename}: missing aux_rec entry")
        if extra:
            errors.append(f"{extra[0]}: aux_rec contains filename not in aux_json")

        for filename in filenames:
            items = aux_rec.get(filename, [])
            rec_stats["total_regions"] += len(items)
            if not items:
                rec_stats["empty_rec_samples"] += 1
                if not args.allow_empty_rec:
                    errors.append(f"{filename}: no OCR regions")
                continue
            width, height = image_sizes.get(filename, (0, 0))
            for item in items:
                bbox = normalize_bbox(item.get("bbox"))
                if bbox is None:
                    rec_stats["invalid_bbox_regions"] += 1
                    continue
                x0, y0, x1, y1 = bbox
                if x1 <= x0 or y1 <= y0:
                    rec_stats["invalid_bbox_regions"] += 1
                if width and height and (x0 < -1 or y0 < -1 or x1 > width + 1 or y1 > height + 1):
                    rec_stats["out_of_bounds_regions"] += 1
                if not str(item.get("text", "")).strip():
                    rec_stats["empty_text_regions"] += 1
    elif args.require_aux_rec:
        errors.append(f"missing aux_rec_pkl: {args.aux_rec_pkl}")

    summary = {
        "samples": len(filenames),
        "images_present": len(image_sizes),
        "errors": len(errors),
        "first_errors": errors[:30],
        **rec_stats,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
