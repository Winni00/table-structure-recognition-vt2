"""Visualize TFLOP aux_rec boxes on their source images."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--aux-rec-pkl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def as_bbox(item: dict[str, Any]) -> list[float] | None:
    bbox = item.get("bbox")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.aux_rec_pkl.open("rb") as f:
        aux_rec = pickle.load(f)

    font = ImageFont.load_default()
    manifest = []
    for idx, (filename, items) in enumerate(sorted(aux_rec.items())[: args.limit]):
        image_path = args.images_dir / filename
        if not image_path.exists():
            manifest.append({"filename": filename, "status": "missing_image"})
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for item_idx, item in enumerate(items):
            bbox = as_bbox(item)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            draw.rectangle([x0, y0, x1, y1], outline=(0, 150, 70), width=2)
            text = str(item.get("text", ""))[:40]
            if text:
                draw.text((x0, max(0, y0 - 10)), text, fill=(0, 90, 40), font=font)
        out_name = f"{idx:02d}_{filename}"
        image.save(args.output_dir / out_name)
        manifest.append(
            {
                "filename": filename,
                "output": str(args.output_dir / out_name),
                "regions": len(items),
            }
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"images": len(manifest), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
