"""Run TableFormer V1 on a prepared public image/html/text-region subset.

This script is designed for public adapted subsets such as:
- FinTabNet public subset prepared for TFLOP
- SynthTabNet public subset prepared for TFLOP

It reuses the same subset artifacts to create a comparable TableFormer V1
reference run:
- `images/`
- `aux_eval.json` with GT HTML
- `end2end_results.pkl` with text-region boxes, IOCR text and other metadata

This is an adaptation path, not an official IBM benchmark reproduction.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
IBM_MODELS_REPO = BASE_DIR / "repo" / "docling-ibm-models"
TFLOP_REPO = BASE_DIR / "repo" / "TFLOP"
LOCAL_TF_ACCURATE_DIR = (
    BASE_DIR / "models" / "docling_ibm" / "tableformer" / "accurate"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "tableformer_public_subset"

if str(IBM_MODELS_REPO) not in sys.path:
    sys.path.insert(0, str(IBM_MODELS_REPO))
if str(TFLOP_REPO) not in sys.path:
    sys.path.insert(0, str(TFLOP_REPO))

from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from tflop.evaluator import TEDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Label written into the output summary, e.g. fintabnet or synthtabnet.",
    )
    parser.add_argument(
        "--subset-root",
        type=Path,
        required=True,
        help="Prepared public subset directory containing images, aux_eval.json and end2end_results.pkl.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for TFPredictor, for example cpu or cuda.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="CPU threads used when device=cpu.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit over the subset. 0 means full prepared subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for predictions, summaries and debug visualizations.",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip writing debug visualizations.",
    )
    return parser.parse_args()


def ensure_html_table_document(text: str) -> str:
    cleaned = text.strip()
    lower = cleaned.lower()
    if "<html" in lower:
        return cleaned
    if "<table" in lower:
        return f"<html><body>{cleaned}</body></html>"
    return f"<html><body><table>{cleaned}</table></body></html>"


def load_predictor_config() -> dict[str, Any]:
    config_path = LOCAL_TF_ACCURATE_DIR / "tm_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["save_dir"] = str(LOCAL_TF_ACCURATE_DIR)
    return config


def load_page_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"))


def text_from_v1_cell(cell: dict[str, Any]) -> str:
    text_boxes = cell.get("text_cell_bboxes") or []
    parts = [str(box.get("token", "")) for box in text_boxes if box.get("token")]
    return "".join(parts)


def html_from_normalized_cells(cells: list[dict[str, Any]], num_rows: int | None, num_cols: int | None) -> str:
    if not cells or not num_rows or not num_cols:
        return ""

    by_start: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in cells:
        by_start[(int(cell["start_row"]), int(cell["start_col"]))] = cell

    header_rows = {
        int(cell["start_row"])
        for cell in cells
        if cell.get("column_header")
    }

    parts: list[str] = ["<table>"]
    if header_rows:
        parts.append("<thead>")
        for row_idx in sorted(header_rows):
            parts.append("<tr>")
            col_idx = 0
            while col_idx < num_cols:
                key = (row_idx, col_idx)
                if key not in by_start:
                    col_idx += 1
                    continue
                cell = by_start[key]
                attrs: list[str] = []
                row_span = int(cell["row_span"])
                col_span = int(cell["col_span"])
                if row_span > 1:
                    attrs.append(f' rowspan="{row_span}"')
                if col_span > 1:
                    attrs.append(f' colspan="{col_span}"')
                parts.append(f"<td{''.join(attrs)}>{cell['text']}</td>")
                col_idx = int(cell["end_col"])
            parts.append("</tr>")
        parts.append("</thead>")

    body_rows = [row_idx for row_idx in range(num_rows) if row_idx not in header_rows]
    if body_rows:
        parts.append("<tbody>")
        for row_idx in body_rows:
            parts.append("<tr>")
            col_idx = 0
            while col_idx < num_cols:
                key = (row_idx, col_idx)
                if key not in by_start:
                    col_idx += 1
                    continue
                cell = by_start[key]
                attrs: list[str] = []
                row_span = int(cell["row_span"])
                col_span = int(cell["col_span"])
                if row_span > 1:
                    attrs.append(f' rowspan="{row_span}"')
                if col_span > 1:
                    attrs.append(f' colspan="{col_span}"')
                parts.append(f"<td{''.join(attrs)}>{cell['text']}</td>")
                col_idx = int(cell["end_col"])
            parts.append("</tr>")
        parts.append("</tbody>")

    parts.append("</table>")
    return "".join(parts)


def normalize_v1_table_output(tf_output: dict[str, Any], table_bbox: list[float]) -> dict[str, Any]:
    predict_details = tf_output["predict_details"]
    tf_responses = tf_output.get("tf_responses", [])

    normalized_cells = []
    for cell in tf_responses:
        normalized_cells.append(
            {
                "bbox": cell.get("bbox"),
                "text": text_from_v1_cell(cell),
                "row_span": cell.get("row_span"),
                "col_span": cell.get("col_span"),
                "start_row": cell.get("start_row_offset_idx"),
                "end_row": cell.get("end_row_offset_idx"),
                "start_col": cell.get("start_col_offset_idx"),
                "end_col": cell.get("end_col_offset_idx"),
                "column_header": cell.get("column_header", False),
                "row_header": cell.get("row_header", False),
                "row_section": cell.get("row_section", False),
            }
        )

    num_rows = predict_details.get("num_rows")
    num_cols = predict_details.get("num_cols")
    rendered_html = html_from_normalized_cells(normalized_cells, num_rows, num_cols)

    return {
        "table_bbox_page": table_bbox,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "cells": normalized_cells,
        "structure": {
            "html": rendered_html,
        },
    }


def render_v1_table_visualization(
    image_path: Path,
    table_bbox: list[float],
    tf_output: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        x1, y1, x2, y2 = table_bbox
        draw.rectangle((x1, y1, x2, y2), outline="pink", width=4)
        for cell in tf_output.get("tf_responses", []):
            bbox = cell.get("bbox")
            if not bbox:
                continue
            draw.rectangle((bbox["l"], bbox["t"], bbox["r"], bbox["b"]), outline="black", width=2)
        canvas.save(output_path)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(payload), indent=2), encoding="utf-8")


def teds_scores(pred_html: str, gt_html: str) -> tuple[float, float]:
    metric_full = TEDS(structure_only=False)
    metric_struct = TEDS(structure_only=True)
    return (
        float(metric_full.evaluate(pred_html, gt_html)),
        float(metric_struct.evaluate(pred_html, gt_html)),
    )


def load_public_subset(subset_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], Path]:
    aux_eval = json.loads((subset_root / "aux_eval.json").read_text(encoding="utf-8"))
    with (subset_root / "end2end_results.pkl").open("rb") as fh:
        aux_rec = pickle.load(fh)
    image_dir = subset_root / "images"
    return aux_eval, aux_rec, image_dir


def token_from_public_region(region: dict[str, Any], token_id: int, text_line_id: int) -> dict[str, Any] | None:
    bbox = region.get("bbox")
    text = str(region.get("text", ""))
    if bbox is None or not text.strip():
        return None
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    return {
        "id": token_id,
        "text": text,
        "bbox": {"l": x0, "t": y0, "r": x1, "b": y1},
        "block_id": 0,
        "text_line_id": text_line_id,
        "indexInLine": 0,
        "confidence": 1.0,
        "word_in_presentation_ltr_order": text,
        "lang": "en",
    }


def prepare_public_page(image_path: Path, regions: list[dict[str, Any]]) -> tuple[dict[str, Any], list[float]]:
    image = load_page_image(image_path)
    height, width = image.shape[:2]
    table_bbox = [0.0, 0.0, float(width), float(height)]

    tokens: list[dict[str, Any]] = []
    for idx, region in enumerate(regions):
        token = token_from_public_region(region, token_id=len(tokens) + 1, text_line_id=idx)
        if token is not None:
            tokens.append(token)

    page = {
        "image": image,
        "png_image_fn": str(image_path),
        "tokens": tokens,
        "width": width,
        "height": height,
        "table_bboxes": [table_bbox],
    }
    return page, table_bbox


def main() -> None:
    args = parse_args()
    aux_eval, aux_rec, image_dir = load_public_subset(args.subset_root)

    filenames = sorted(aux_eval.keys())
    if args.limit > 0:
        filenames = filenames[: args.limit]

    predictor = TFPredictor(load_predictor_config(), device=args.device, num_threads=args.num_threads)

    teds_values: list[float] = []
    teds_s_values: list[float] = []
    per_table: list[dict[str, Any]] = []

    for sample_index, filename in enumerate(filenames):
        image_path = image_dir / filename
        gt_html = ensure_html_table_document(aux_eval[filename]["html"])
        iocr_page, table_bbox = prepare_public_page(image_path, aux_rec.get(filename, []))

        raw_results = predictor.multi_table_predict(
            iocr_page,
            deepcopy([table_bbox]),
            do_matching=True,
            correct_overlapping_cells=False,
            sort_row_col_indexes=True,
        )
        raw_output = raw_results[0]
        normalized_output = normalize_v1_table_output(raw_output, table_bbox)
        pred_html = ensure_html_table_document(normalized_output["structure"].get("html") or "")
        teds, teds_s = teds_scores(pred_html, gt_html)
        teds_values.append(teds)
        teds_s_values.append(teds_s)

        viz_path = None
        if not args.no_viz:
            viz_path = args.output_dir / "viz" / f"{Path(filename).stem}.png"
            render_v1_table_visualization(image_path, table_bbox, raw_output, viz_path)

        sample_payload = {
            "sample": Path(filename).stem,
            "dataset": args.dataset_name,
            "split": "public_subset",
            "sample_index": sample_index,
            "image_path": str(image_path),
            "model_version": "tableformer_v1",
            "weights_dir": str(LOCAL_TF_ACCURATE_DIR),
            "table_bboxes": [table_bbox],
            "raw_output": raw_results,
            "normalized_output": [normalized_output],
            "ground_truth": {
                "gt_html": gt_html,
                "type": aux_eval[filename].get("type"),
            },
            "evaluation_result": {
                "teds": teds,
                "teds_s": teds_s,
            },
            "debug_visualizations": [] if viz_path is None else [str(viz_path)],
        }
        write_json(args.output_dir / f"{Path(filename).stem}.pred.json", sample_payload)

        per_table.append(
            {
                "sample": Path(filename).stem,
                "filename": filename,
                "type": aux_eval[filename].get("type"),
                "teds": teds,
                "teds_s": teds_s,
                "pred_path": str(args.output_dir / f"{Path(filename).stem}.pred.json"),
            }
        )

    summary = {
        "dataset": args.dataset_name,
        "split": "public_subset",
        "mode": "adaptation",
        "subset_size": len(filenames),
        "device": args.device,
        "weights_dir": str(LOCAL_TF_ACCURATE_DIR),
        "source_repo": str(IBM_MODELS_REPO),
        "source_subset_root": str(args.subset_root),
        "mean_teds": statistics.mean(teds_values) if teds_values else None,
        "mean_teds_s": statistics.mean(teds_s_values) if teds_s_values else None,
        "notes": [
            "This is a public adaptation path using the same prepared subset used for TFLOP reference runs.",
            "Ground-truth HTML comes from aux_eval.json.",
            "TableFormer input tokens are built from the public text-region pickle.",
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "per_table.json", per_table)

    print("TableFormer public subset reference run finished.")
    print(f"Dataset: {args.dataset_name}")
    print(f"Subset size: {len(filenames)}")
    print(f"Mean TEDS: {summary['mean_teds']}")
    print(f"Mean TEDS-S: {summary['mean_teds_s']}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
