#!/usr/bin/env python3
"""Build FinTabNet train/validation data in TFLOP training format."""

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

from build_tflop_fintabnet_annotation_aux import (  # noqa: E402
    cell_bbox_to_crop_coords,
    rich_text,
    sanitize_filename,
)
from build_tflop_fintabnet_smoke_aux import (  # noqa: E402
    render_with_mediabox,
    table_bbox_to_image_coords,
)


TFLOP_ROOT = BASE_DIR / "repo" / "TFLOP"
KAGGLE_ROOT = BASE_DIR / "data" / "fintabnet_1.0.0_kaggle_full" / "fintabnet"
DEFAULT_TRAIN_JSONL = KAGGLE_ROOT / "FinTabNet_1.0.0_cell_train.jsonl"
DEFAULT_VAL_JSONL = KAGGLE_ROOT / "FinTabNet_1.0.0_cell_val.jsonl"
DEFAULT_PDF_DIR = KAGGLE_ROOT / "pdf"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "tflop_fintabnet_train_smoke_1k"


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


def split_top_level_rows(tokens: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    current: list[str] = []
    depth = 0
    for token in tokens:
        if token == "<tr>":
            if depth == 0:
                current = []
            depth += 1
        if depth > 0:
            current.append(token)
        if token == "</tr>" and depth > 0:
            depth -= 1
            if depth == 0 and current:
                rows.append(current)
                current = []
    return rows


def to_tflop_section_tokens(tokens: list[str], section_mode: str = "empty_thead") -> list[str]:
    """Wrap direct FinTabNet rows so the TFLOP OTSL converter accepts them."""
    tokens = strip_table_tokens(tokens)
    if "<thead>" in tokens and "</thead>" in tokens:
        return tokens
    if section_mode == "first_row_thead":
        rows = split_top_level_rows(tokens)
        if rows:
            thead_tokens = [token for row in rows[:1] for token in row]
            tbody_tokens = [token for row in rows[1:] for token in row]
            return ["<thead>"] + thead_tokens + ["</thead>", "<tbody>"] + tbody_tokens + ["</tbody>"]
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


def build_entry(
    row_idx: int,
    row: dict[str, Any],
    split: str,
    pdf_dir: Path,
    image_dir: Path,
    render_scale: float,
    max_cells: int,
    convert_html_to_otsl,
    otsl_map: dict[str, str],
    section_mode: str,
) -> dict[str, Any] | None:
    cells = row["html"]["cells"]
    if len(cells) > max_cells:
        return None

    pdf_path = pdf_dir / row["filename"]
    if not pdf_path.exists() or not row.get("bbox"):
        return None

    image, mediabox = render_with_mediabox(pdf_path, render_scale)
    crop_left, crop_top, crop_right, crop_bottom = table_bbox_to_image_coords(
        row["bbox"], mediabox, render_scale
    )
    image_crop = image[crop_top:crop_bottom, crop_left:crop_right]
    if image_crop.size == 0 or image_crop.shape[0] == 0 or image_crop.shape[1] == 0:
        return None

    crop_size = (image_crop.shape[1], image_crop.shape[0])
    crop_bboxes: list[list[float] | None] = []
    for cell in cells:
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

    org_html = to_tflop_section_tokens(row["html"]["structure"]["tokens"], section_mode)
    try:
        otsl_seq, num_rows, num_cols = convert_html_to_otsl(org_html, otsl_map)
    except Exception:
        return None

    gold_coord = [bbox_string(cell, crop_bbox) for cell, crop_bbox in zip(cells, crop_bboxes)]
    dr_coord = build_dr_coord(cells, crop_bboxes)
    if not dr_coord:
        return None

    out_name = f"fintabnet_{split}_{row_idx:06d}_{sanitize_filename(row['table_id'])}.png"
    from PIL import Image

    Image.fromarray(image_crop).save(image_dir / out_name)
    return {
        "file_name": out_name,
        "dr_coord": dr_coord,
        "gold_coord": gold_coord,
        "org_html": org_html,
        "otsl_seq": otsl_seq,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "split": split,
        "source_filename": row["filename"],
        "table_id": row["table_id"],
    }


def build_split(
    jsonl_path: Path,
    split: str,
    limit: int,
    pdf_dir: Path,
    image_dir: Path,
    render_scale: float,
    max_cells: int,
    convert_html_to_otsl,
    otsl_map: dict[str, str],
    section_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    entries: list[dict[str, Any]] = []
    stats = {"seen": 0, "kept": 0, "skipped": 0}
    with jsonl_path.open(encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            stats["seen"] += 1
            row = json.loads(line)
            entry = build_entry(
                row_idx,
                row,
                split,
                pdf_dir,
                image_dir,
                render_scale,
                max_cells,
                convert_html_to_otsl,
                otsl_map,
                section_mode,
            )
            if entry is None:
                stats["skipped"] += 1
                continue
            entries.append(entry)
            stats["kept"] += 1
            if len(entries) >= limit:
                break
    return entries, stats


def write_jsonl(path: Path, entries: list[dict[str, Any]], split: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps({**entry, "split": split}, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--max-cells", type=int, default=140)
    parser.add_argument("--render-scale", type=float, default=2.0)
    parser.add_argument("--bbox-token-cnt", type=int, default=640)
    parser.add_argument("--max-length", type=int, default=1376)
    parser.add_argument(
        "--section-mode",
        choices=["empty_thead", "first_row_thead"],
        default="empty_thead",
        help="How to wrap FinTabNet direct table rows before OTSL conversion.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.force:
        shutil.rmtree(args.out_dir)

    meta_dir = args.out_dir / "meta_data"
    train_img_dir = args.out_dir / "images" / "train"
    val_img_dir = args.out_dir / "images" / "validation"
    meta_dir.mkdir(parents=True, exist_ok=True)
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    convert_html_to_otsl = load_convert_html_to_otsl()
    otsl_map = load_otsl_map()

    train_entries, train_stats = build_split(
        args.train_jsonl,
        "train",
        args.train_samples,
        args.pdf_dir,
        train_img_dir,
        args.render_scale,
        args.max_cells,
        convert_html_to_otsl,
        otsl_map,
        args.section_mode,
    )
    val_entries, val_stats = build_split(
        args.val_jsonl,
        "validation",
        args.val_samples,
        args.pdf_dir,
        val_img_dir,
        args.render_scale,
        args.max_cells,
        convert_html_to_otsl,
        otsl_map,
        args.section_mode,
    )

    if not train_entries or not val_entries:
        raise RuntimeError("No train or validation entries generated")

    train_path = meta_dir / "dataset_train.jsonl"
    val_path = meta_dir / "dataset_validation.jsonl"
    write_jsonl(train_path, train_entries, "train")
    write_jsonl(val_path, val_entries, "validation")

    data_config_path = args.out_dir / "data_config.yaml"
    data_config_path.write_text(
        "\n".join(
            [
                f"image_path: {args.out_dir / 'images'}",
                f"meta_data_path: {meta_dir}",
                "input_size:",
                "  height: 768",
                "  width: 768",
                "window_size: 8",
                "align_along_axis: False",
                f"max_length: {args.max_length}",
                f"bbox_token_cnt: {args.bbox_token_cnt}",
                "use_cell_bbox: False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "train_entries": len(train_entries),
        "validation_entries": len(val_entries),
        "train_stats": train_stats,
        "validation_stats": val_stats,
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "pdf_dir": str(args.pdf_dir),
        "out_dir": str(args.out_dir),
        "data_config": str(data_config_path),
        "max_cells": args.max_cells,
        "render_scale": args.render_scale,
        "bbox_token_cnt": args.bbox_token_cnt,
        "max_length": args.max_length,
        "adapter": "FinTabNet annotation cell boxes/text converted to TFLOP dr_coord/gold_coord",
        "section_mode": args.section_mode,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
