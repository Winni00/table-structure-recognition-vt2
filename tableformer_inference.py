"""TableFormer inference runner for official samples and selected hard tables.

This script runs the original TableFormer predictor (`TFPredictor`) from the
local `docling-ibm-models` clone and supports two practical input modes:

1. Official repo mode:
   - page image
   - IOCR/parse JSON with tokens
   - table bounding boxes

2. Hard-table dataset mode:
   - table crop image from `selected_hard_tables.json`
   - the full crop is treated as a single table region
   - no IOCR token matching is available unless extra metadata is provided

The hard-table mode is useful for model-to-model comparison because it gives
both TableFormer variants the same table crop inputs and a shared JSON export.
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
IBM_MODELS_REPO = BASE_DIR / "repo" / "docling-ibm-models"
LOCAL_TF_ACCURATE_DIR = BASE_DIR / "models" / "docling_ibm" / "tableformer" / "accurate"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "tableformer_inference"
DEFAULT_HARD_TABLES_JSON = (
    BASE_DIR / "data" / "paper_collection_meta" / "selected_hard_tables.json"
)

# Make the cloned repo importable so we can use the local implementation directly.
if str(IBM_MODELS_REPO) not in sys.path:
    sys.path.insert(0, str(IBM_MODELS_REPO))

from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor


# These samples mirror the official repo tests and are a good first validation target.
OFFICIAL_SAMPLE_SPECS = [
    {
        "name": "ADS.2007.page_123",
        "json_path": IBM_MODELS_REPO
        / "tests"
        / "test_data"
        / "samples"
        / "ADS.2007.page_123.png_iocr.parse_format.json",
        "image_path": IBM_MODELS_REPO / "tests" / "test_data" / "samples" / "ADS.2007.page_123.png",
        "table_bboxes": [[178, 748, 1061, 976], [177, 1163, 1062, 1329]],
    },
    {
        "name": "PHM.2013.page_30",
        "json_path": IBM_MODELS_REPO
        / "tests"
        / "test_data"
        / "samples"
        / "PHM.2013.page_30.png_iocr.parse_format.json",
        "image_path": IBM_MODELS_REPO / "tests" / "test_data" / "samples" / "PHM.2013.page_30.png",
        "table_bboxes": [[100, 186, 1135, 525]],
    },
    {
        "name": "empty_iocr",
        "json_path": IBM_MODELS_REPO / "tests" / "test_data" / "samples" / "empty_iocr.png.json",
        "image_path": IBM_MODELS_REPO / "tests" / "test_data" / "samples" / "empty_iocr.png",
        "table_bboxes": [[178, 748, 1061, 976], [177, 1163, 1062, 1329]],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Direct TableFormer inference via docling-ibm-models/TFPredictor "
            "with local accurate weights."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["official", "custom", "hard_tables"],
        default="official",
        help="Run official repo samples, one custom input, or the selected hard tables dataset.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Custom page image path. Required in custom mode.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Custom IOCR/parse JSON path. Required in custom mode.",
    )
    parser.add_argument(
        "--bboxes",
        type=str,
        help='Custom table bboxes as JSON string, e.g. "[[100, 200, 900, 700]]".',
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_HARD_TABLES_JSON,
        help="Path to the selected hard tables metadata JSON used in hard_tables mode.",
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
        "--no-cell-matching",
        action="store_true",
        help="Disable token matching and keep the raw predicted cell boxes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for prediction outputs.",
    )
    return parser.parse_args()


def load_predictor_config() -> dict:
    """Load the official TableFormer config and redirect it to local weights."""
    config_path = LOCAL_TF_ACCURATE_DIR / "tm_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    # The predictor expects the checkpoint directory in model.save_dir.
    config["model"]["save_dir"] = str(LOCAL_TF_ACCURATE_DIR)
    return config


def load_page_image(image_path: Path) -> np.ndarray:
    """Load a page image as an RGB numpy array."""
    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"))


def load_iocr_page(json_path: Path) -> dict:
    """Load the page payload from the repo-style IOCR/parse JSON."""
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if "pages" in raw:
        return raw["pages"][0]
    return raw


def build_custom_spec(args: argparse.Namespace) -> dict:
    """Create one custom sample spec from CLI arguments."""
    if args.image is None or args.json is None or args.bboxes is None:
        raise ValueError("Custom mode requires --image, --json, and --bboxes.")
    return {
        "name": args.image.stem,
        "json_path": args.json,
        "image_path": args.image,
        "table_bboxes": json.loads(args.bboxes),
    }


def build_hard_table_specs(dataset_json: Path) -> list[dict]:
    """Create sample specs from the selected hard tables metadata file."""
    with dataset_json.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    specs = []
    for row in rows:
        input_png = Path(row["input_png"])
        with Image.open(input_png) as img:
            width, height = img.size

        # In hard-table mode the provided image is already a table crop,
        # so the whole image becomes the single inference region.
        specs.append(
            {
                "name": row["table_id"],
                "image_path": input_png,
                "json_path": None,
                "table_bboxes": [[0, 0, width, height]],
                "metadata": row,
            }
        )
    return specs


def prepare_iocr_page(spec: dict) -> dict:
    """Build the in-memory payload expected by `TFPredictor.multi_table_predict`."""
    if spec["json_path"] is None:
        # Hard-table mode does not come with IOCR/parse JSON, so we create the
        # smallest valid page payload expected by the predictor.
        image = load_page_image(spec["image_path"])
        height, width = image.shape[:2]
        return {
            "image": image,
            "png_image_fn": str(spec["image_path"]),
            "table_bboxes": deepcopy(spec["table_bboxes"]),
            "tokens": [],
            "width": width,
            "height": height,
        }

    page = load_iocr_page(spec["json_path"])
    page["image"] = load_page_image(spec["image_path"])
    page["png_image_fn"] = str(spec["image_path"])
    page["table_bboxes"] = deepcopy(spec["table_bboxes"])
    page.setdefault("tokens", [])

    # Some custom inputs may not contain width/height in JSON, so derive them from the image.
    if "width" not in page or "height" not in page:
        height, width = page["image"].shape[:2]
        page["width"] = width
        page["height"] = height
    return page


def make_json_safe(value):
    """Convert numpy types to plain Python so the output can be serialized."""
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


def write_prediction(output_dir: Path, sample_name: str, sample_output: dict) -> None:
    """Persist one sample prediction bundle to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_name}.pred.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(sample_output), f, indent=2)


def to_v1_otsl_text(rs_seq: list[str]) -> str:
    """Convert the V1 row/structure token list to an OTSL-like string."""
    return " ".join(f"<{token}>" for token in rs_seq)


def to_v1_html_text(html_seq: list[str]) -> str:
    """Wrap the raw V1 HTML token sequence into a complete table tag."""
    return f"<table>{''.join(html_seq)}</table>"


def normalize_v1_table_output(tf_output: dict, table_bbox: list[float]) -> dict:
    """Project native V1 output to a simpler model-agnostic table structure."""
    predict_details = tf_output["predict_details"]
    prediction = predict_details.get("prediction", {})
    rs_seq = prediction.get("rs_seq", [])
    html_seq = prediction.get("html_seq", [])
    tf_responses = tf_output.get("tf_responses", [])

    normalized_cells = []
    for cell in tf_responses:
        normalized_cells.append(
            {
                "bbox": cell.get("bbox"),
                "text": cell.get("bbox", {}).get("token", ""),
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

    return {
        "table_bbox_page": table_bbox,
        "num_rows": predict_details.get("num_rows"),
        "num_cols": predict_details.get("num_cols"),
        "cells": normalized_cells,
        "structure": {
            "otsl_tokens": rs_seq,
            "otsl_text": to_v1_otsl_text(rs_seq),
            "html": to_v1_html_text(html_seq) if html_seq else "",
        },
    }


def render_v1_table_visualization(
    image_path: Path,
    table_bbox: list[float],
    tf_output: dict,
    output_path: Path,
) -> None:
    """Draw the predicted table region and cell boxes on the source image."""
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

            color = "black"
            width = 2
            if cell.get("column_header"):
                color = "blue"
                width = 3
            elif cell.get("row_header"):
                color = "magenta"
                width = 3
            elif cell.get("row_section"):
                color = "brown"
                width = 3

            draw.rectangle((bbox["l"], bbox["t"], bbox["r"], bbox["b"]), outline=color, width=width)

        canvas.save(output_path)


def main() -> None:
    args = parse_args()
    config = load_predictor_config()
    predictor = TFPredictor(config, device=args.device, num_threads=args.num_threads)

    if args.mode == "official":
        specs = OFFICIAL_SAMPLE_SPECS
    elif args.mode == "custom":
        specs = [build_custom_spec(args)]
    else:
        specs = build_hard_table_specs(args.dataset_json)

    # Hard-table crops do not include IOCR token payloads, so matching must stay off there.
    do_matching = False if args.mode == "hard_tables" else not args.no_cell_matching

    summary = []
    for spec in specs:
        # Build the exact page-like input object used by the original predictor.
        iocr_page = prepare_iocr_page(spec)

        # The predictor mutates bbox lists internally, so we pass a defensive copy.
        results = predictor.multi_table_predict(
            iocr_page,
            deepcopy(spec["table_bboxes"]),
            do_matching=do_matching,
            correct_overlapping_cells=False,
            sort_row_col_indexes=True,
        )

        viz_paths = []
        normalized_results = []
        for table_idx, tf_output in enumerate(results):
            table_bbox = spec["table_bboxes"][table_idx]
            normalized_results.append(normalize_v1_table_output(tf_output, table_bbox))

            viz_path = args.output_dir / "viz" / f"{spec['name']}.table_{table_idx}.png"
            render_v1_table_visualization(spec["image_path"], table_bbox, tf_output, viz_path)
            viz_paths.append(str(viz_path))

        sample_output = {
            "sample": spec["name"],
            "model_version": "tableformer",
            "weights_dir": str(LOCAL_TF_ACCURATE_DIR),
            "do_cell_matching": do_matching,
            "image_path": str(spec["image_path"]),
            "json_path": None if spec["json_path"] is None else str(spec["json_path"]),
            "table_bboxes": spec["table_bboxes"],
            # Keep the native predictor output untouched for later fair comparison.
            "raw_output": results,
            # Keep a legacy alias for compatibility with older inspection scripts.
            "results": results,
            "normalized_output": normalized_results,
            "debug_visualizations": viz_paths,
            "evaluation_result": None,
        }
        if "metadata" in spec:
            sample_output["dataset_metadata"] = spec["metadata"]
            sample_output["ground_truth"] = {
                "gt_html": spec["metadata"].get("gt_html"),
                "gt_xml": spec["metadata"].get("gt_xml"),
            }
        write_prediction(args.output_dir, spec["name"], sample_output)

        summary.append(
            {
                "sample": spec["name"],
                "tables_requested": len(spec["table_bboxes"]),
                "tables_predicted": len(results),
                "debug_visualizations": viz_paths,
                "output_file": str(args.output_dir / f"{spec['name']}.pred.json"),
            }
        )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Direct TableFormer inference finished.")
    print(f"Weights: {LOCAL_TF_ACCURATE_DIR}")
    print(f"Output dir: {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
