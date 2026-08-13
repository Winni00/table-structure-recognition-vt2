#!/usr/bin/env python3
"""Build a small FinTabNet dataset in TFLOP training format.

The public TFLOP training code expects PubTabNet-like metadata files with
OTSL labels, gold cell coordinates, and detection/recognition box groups.  This
script converts our FinTabNet Kaggle annotation input into that format for a
small overfit sanity check.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
sys.path.insert(0, str(BASE_DIR))

from build_tflop_fintabnet_annotation_aux import (
    cell_bbox_to_crop_coords,
    rich_text,
    sanitize_filename,
)
from build_tflop_fintabnet_smoke_aux import (
    render_with_mediabox,
    table_bbox_to_image_coords,
)


TFLOP_ROOT = BASE_DIR / "repo" / "TFLOP"
FTN_JSONL = BASE_DIR / "data" / "fintabnet_kaggle" / "FinTabNet_1.0.0_cell_val_excluding_28.jsonl"
FTN_PDF_DIR = BASE_DIR / "data" / "fintabnet_kaggle" / "pdfs"
FTN_RUN_IMAGES = BASE_DIR / "results" / "tflop_fintabnet_full_annotation_excl_problem_pages" / "images"
OUT_DIR = BASE_DIR / "results" / "tflop_fintabnet_overfit100"


def load_convert_html_to_otsl():
    module_path = TFLOP_ROOT / "dataset" / "preprocess_data_utils.py"
    spec = importlib.util.spec_from_file_location("tflop_preprocess_data_utils", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.convert_html_to_otsl


def load_otsl_map() -> dict[str, str]:
    config_path = TFLOP_ROOT / "dataset" / "data_preprocessing_config.json"
    data = json.loads(config_path.read_text())
    return data["OTSL_TAG"]


def strip_table_tokens(tokens: list[str]) -> list[str]:
    tokens = list(tokens)
    if tokens and tokens[0] == "<table>":
        tokens = tokens[1:]
    if tokens and tokens[-1] == "</table>":
        tokens = tokens[:-1]
    return tokens


def to_tflop_section_tokens(tokens: list[str]) -> list[str]:
    """Make FinTabNet direct-row HTML acceptable to TFLOP's OTSL converter."""
    tokens = strip_table_tokens(tokens)
    if "<thead>" in tokens and "</thead>" in tokens:
        return tokens
    return ["<thead>", "</thead>", "<tbody>"] + tokens + ["</tbody>"]


def bbox_string(cell: dict[str, Any], crop_bbox: list[float] | None) -> str:
    text = rich_text(cell.get("tokens", []))
    if crop_bbox is None or not text:
        return "-1.0 -1.0 -1.0 -1.0 1 "
    coords = ["%.2f" % c for c in crop_bbox]
    return " ".join(coords + ["2", text])


def build_dr_coord(cells: list[dict[str, Any]], crop_bboxes: list[list[float] | None]) -> dict[int, list[Any]]:
    dr_coord: dict[int, list[Any]] = {}
    group_idx = 0
    for cell_idx, (cell, crop_bbox) in enumerate(zip(cells, crop_bboxes)):
        text = rich_text(cell.get("tokens", []))
        if crop_bbox is None or not text:
            continue
        dr_coord[group_idx] = [[crop_bbox], cell_idx, text]
        group_idx += 1
    return dr_coord


def make_entry(
    row_idx: int,
    row: dict[str, Any],
    render_scale: float,
    convert_html_to_otsl,
    otsl_map: dict[str, str],
) -> tuple[dict[str, Any], str] | None:
    out_name = f"fintabnet_{row_idx:05d}_{sanitize_filename(row['table_id'])}.png"
    source_image = FTN_RUN_IMAGES / out_name
    if not source_image.exists():
        return None

    pdf_path = FTN_PDF_DIR / row["filename"]
    if not pdf_path.exists() or not row.get("bbox"):
        return None

    image, mediabox = render_with_mediabox(pdf_path, render_scale)
    crop_left, crop_top, crop_right, crop_bottom = table_bbox_to_image_coords(
        row["bbox"], mediabox, render_scale
    )
    crop_size = (crop_right - crop_left, crop_bottom - crop_top)
    if crop_size[0] <= 0 or crop_size[1] <= 0 or image.size == 0:
        return None

    crop_bboxes: list[list[float] | None] = []
    for cell in row["html"]["cells"]:
        bbox = cell.get("bbox")
        if not bbox or len(bbox) != 4:
            crop_bboxes.append(None)
            continue
        crop_bboxes.append(
            cell_bbox_to_crop_coords(
                bbox,
                mediabox,
                render_scale,
                (crop_left, crop_top),
                crop_size,
            )
        )

    org_html = to_tflop_section_tokens(row["html"]["structure"]["tokens"])
    otsl_seq, num_rows, num_cols = convert_html_to_otsl(org_html, otsl_map)
    gold_coord = [
        bbox_string(cell, crop_bbox)
        for cell, crop_bbox in zip(row["html"]["cells"], crop_bboxes)
    ]
    dr_coord = build_dr_coord(row["html"]["cells"], crop_bboxes)
    if not dr_coord:
        return None

    entry = {
        "file_name": out_name,
        "dr_coord": dr_coord,
        "gold_coord": gold_coord,
        "org_html": org_html,
        "otsl_seq": otsl_seq,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "split": "train",
        "source_filename": row["filename"],
        "table_id": row["table_id"],
    }
    return entry, out_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-cells", type=int, default=140)
    parser.add_argument("--render-scale", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_html_to_otsl = load_convert_html_to_otsl()
    otsl_map = load_otsl_map()

    meta_dir = args.out_dir / "meta_data"
    train_img_dir = args.out_dir / "images" / "train"
    val_img_dir = args.out_dir / "images" / "validation"
    meta_dir.mkdir(parents=True, exist_ok=True)
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    image_names: list[str] = []
    rows = [json.loads(line) for line in FTN_JSONL.open(encoding="utf-8")]
    for row_idx, row in enumerate(rows):
        if len(row["html"]["cells"]) > args.max_cells:
            continue
        built = make_entry(row_idx, row, args.render_scale, convert_html_to_otsl, otsl_map)
        if built is None:
            continue
        entry, image_name = built
        entries.append(entry)
        image_names.append(image_name)
        if len(entries) >= args.num_samples:
            break

    if not entries:
        raise RuntimeError("No training entries were generated")

    # Overfit sanity: train and validate on the same small set.
    for image_name in image_names:
        source = FTN_RUN_IMAGES / image_name
        shutil.copy2(source, train_img_dir / image_name)
        shutil.copy2(source, val_img_dir / image_name)

    train_path = meta_dir / "dataset_train.jsonl"
    val_path = meta_dir / "dataset_validation.jsonl"
    with train_path.open("w", encoding="utf-8") as f_train, val_path.open("w", encoding="utf-8") as f_val:
        for entry in entries:
            f_train.write(json.dumps({**entry, "split": "train"}, ensure_ascii=False) + "\n")
            f_val.write(json.dumps({**entry, "split": "validation"}, ensure_ascii=False) + "\n")

    data_config = {
        "image_path": str(args.out_dir / "images"),
        "meta_data_path": str(meta_dir),
        "input_size": {"height": 768, "width": 768},
        "window_size": 8,
        "align_along_axis": False,
        "max_length": 2700,
        "bbox_token_cnt": 864,
        "use_cell_bbox": False,
    }
    data_config_path = args.out_dir / "data_config.yaml"
    data_config_path.write_text(
        "\n".join(
            [
                f"image_path: {data_config['image_path']}",
                f"meta_data_path: {data_config['meta_data_path']}",
                "input_size:",
                "  height: 768",
                "  width: 768",
                "window_size: 8",
                "align_along_axis: False",
                "max_length: 2700",
                "bbox_token_cnt: 864",
                "use_cell_bbox: False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "samples": len(entries),
        "max_cells": args.max_cells,
        "output_dir": str(args.out_dir),
        "data_config": str(data_config_path),
        "train_jsonl": str(train_path),
        "validation_jsonl": str(val_path),
        "note": "Overfit sanity dataset: train and validation contain the same entries.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
