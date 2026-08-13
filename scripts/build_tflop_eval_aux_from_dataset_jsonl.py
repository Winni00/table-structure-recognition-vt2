#!/usr/bin/env python3
"""Build TFLOP test/eval aux files from a TFLOP dataset_*.jsonl file."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
TFLOP_ROOT = BASE_DIR / "repo" / "TFLOP"
sys.path.insert(0, str(TFLOP_ROOT))

from tflop.datamodule.preprocess.common_utils import (  # noqa: E402
    convert_gold_coords,
    generate_filled_html,
)


def html_from_dataset_entry(entry: dict[str, Any]) -> str:
    gold_coords = convert_gold_coords(entry["gold_coord"])
    table_inner = generate_filled_html(
        gold_text_list=gold_coords["text"],
        is_cell_filled=gold_coords["isFilled"],
        org_html_list=entry["org_html"],
    )
    if "<table" in table_inner:
        return table_inner
    return f"<html><body><table>{table_inner}</table></body></html>"


def rec_from_dataset_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rec_items: list[dict[str, Any]] = []
    for key in sorted(entry["dr_coord"], key=lambda x: int(x)):
        bbox_group, _cell_idx, text = entry["dr_coord"][key]
        for bbox_idx, bbox in enumerate(bbox_group):
            rec_items.append(
                {
                    "bbox": [float(x) for x in bbox],
                    "text": text if bbox_idx == 0 else "",
                    "score": 1.0,
                }
            )
    return rec_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--source-image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_out = args.output_dir / "images"
    image_out.mkdir(parents=True, exist_ok=True)

    aux_json: dict[str, dict[str, str]] = {}
    aux_rec: dict[str, list[dict[str, Any]]] = {}
    copied = 0
    with args.dataset_jsonl.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            filename = entry["file_name"]
            src = args.source_image_dir / filename
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, image_out / filename)
            copied += 1
            aux_json[filename] = {
                "html": html_from_dataset_entry(entry),
                "type": "simple",
            }
            aux_rec[filename] = rec_from_dataset_entry(entry)

    (args.output_dir / "aux.json").write_text(
        json.dumps(aux_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.output_dir / "aux_rec.pkl").open("wb") as f:
        pickle.dump(aux_rec, f)

    summary = {
        "samples": len(aux_json),
        "images_copied": copied,
        "dataset_jsonl": str(args.dataset_jsonl),
        "source_image_dir": str(args.source_image_dir),
        "output_dir": str(args.output_dir),
        "total_text_regions": sum(len(v) for v in aux_rec.values()),
        "empty_text_region_samples": sum(1 for v in aux_rec.values() if not v),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
