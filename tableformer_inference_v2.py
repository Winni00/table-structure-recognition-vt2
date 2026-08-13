"""TableFormerV2 inference runner for official samples and selected hard tables.

This script uses the explicit `tableformer_v2` model path from the local
`docling-ibm-models` clone. Unlike the original TableFormer path, V2 works
directly on table crops and produces:
- generated OTSL tokens
- predicted cell bounding boxes normalized to the crop

It supports:
1. official repo sample pages with provided table boxes
2. one custom page image plus boxes
3. the selected hard tables dataset, where each provided PNG is already a crop
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.document import DoclingDocument, TableCell, TableData

BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
IBM_MODELS_REPO = BASE_DIR / "repo" / "docling-ibm-models"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "tableformer_v2_inference"
DEFAULT_V2_LOCAL_DIR = BASE_DIR / "models" / "docling_ibm" / "tableformer_v2"
DEFAULT_HARD_TABLES_JSON = (
    BASE_DIR / "data" / "paper_collection_meta" / "selected_hard_tables.json"
)

# Make the cloned repo importable so we can use the local implementation directly.
if str(IBM_MODELS_REPO) not in sys.path:
    sys.path.insert(0, str(IBM_MODELS_REPO))

from docling_ibm_models.tableformer_v2 import TableFormerV2


OFFICIAL_SAMPLE_SPECS = [
    {
        "name": "ADS.2007.page_123",
        "image_path": IBM_MODELS_REPO / "tests" / "test_data" / "samples" / "ADS.2007.page_123.png",
        "table_bboxes": [[178, 748, 1061, 976], [177, 1163, 1062, 1329]],
    },
    {
        "name": "PHM.2013.page_30",
        "image_path": IBM_MODELS_REPO / "tests" / "test_data" / "samples" / "PHM.2013.page_30.png",
        "table_bboxes": [[100, 186, 1135, 525]],
    },
    {
        "name": "empty_iocr",
        "image_path": IBM_MODELS_REPO / "tests" / "test_data" / "samples" / "empty_iocr.png",
        "table_bboxes": [[178, 748, 1061, 976], [177, 1163, 1062, 1329]],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Direct TableFormer V2 inference on table crops. "
            "Requires a local V2 artifact path or a repo id."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["official", "custom", "hard_tables"],
        default="official",
        help="Run official repo samples, one custom input, or the selected hard tables dataset.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_V2_LOCAL_DIR,
        help="Local TableFormer V2 artifact path containing model and tokenizer files.",
    )
    parser.add_argument(
        "--artifact-repo-id",
        default=None,
        help="Optional Hugging Face repo id for V2 if you do not use a local artifact path.",
    )
    parser.add_argument(
        "--artifact-revision",
        default=None,
        help="Optional revision for the V2 artifact path or repo id.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Custom page image path. Required in custom mode.",
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
        help="Torch device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
        help="Square resize used before feeding the crop into V2.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum autoregressive sequence length.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for prediction outputs.",
    )
    return parser.parse_args()


def load_tokenizer(path: str, revision: str | None = None):
    """Load a tokenizer either from a local artifact folder or from Hugging Face."""
    tokenizer_file = Path(path) / "tokenizer.json"
    if tokenizer_file.exists():
        backend_tokenizer = Tokenizer.from_file(str(tokenizer_file))
        return PreTrainedTokenizerFast(
            tokenizer_object=backend_tokenizer,
            bos_token="<start>",
            eos_token="<end>",
            pad_token="<pad>",
            unk_token="[UNK]",
        )

    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<start>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<end>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    return tokenizer


def build_custom_spec(args: argparse.Namespace) -> dict:
    """Create one custom sample spec from CLI arguments."""
    if args.image is None or args.bboxes is None:
        raise ValueError("Custom mode requires --image and --bboxes.")
    return {
        "name": args.image.stem,
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
                "table_bboxes": [[0, 0, width, height]],
                "metadata": row,
            }
        )
    return specs


def load_model_and_tokenizer(args: argparse.Namespace):
    """Resolve a V2 model either from a local artifact folder or from a repo id."""
    if args.artifact_path.exists():
        model = TableFormerV2.from_pretrained(
            str(args.artifact_path),
            revision=args.artifact_revision,
        )
        tokenizer = load_tokenizer(str(args.artifact_path), revision=args.artifact_revision)
        return model, tokenizer, str(args.artifact_path)

    if args.artifact_repo_id is None:
        raise FileNotFoundError(
            f"Local V2 artifact path not found: {args.artifact_path}. "
            "Provide --artifact-path to a downloaded V2 checkpoint or set --artifact-repo-id."
        )

    model = TableFormerV2.from_pretrained(
        args.artifact_repo_id,
        revision=args.artifact_revision,
    )
    tokenizer = load_tokenizer(args.artifact_repo_id, revision=args.artifact_revision)
    return model, tokenizer, args.artifact_repo_id


def make_transform(image_size: int):
    """Use the same image preprocessing recipe as the official V2 tests."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def make_json_safe(value):
    """Convert tensors and numpy values to plain Python for JSON serialization."""
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_prediction(output_dir: Path, sample_name: str, sample_output: dict) -> None:
    """Persist one sample prediction bundle to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_name}.pred.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(sample_output), f, indent=2)


def extract_otsl_tokens(otsl_text: str) -> list[str]:
    """Extract normalized OTSL tags like `fcel` and `nl` from a generated string."""
    return [token.strip("<>") for token in re.findall(r"<[^>]+>", otsl_text)]


def build_table_data_from_otsl(otsl_text: str) -> TableData:
    """Build empty-text table cells from a structural OTSL sequence."""
    otsl_tokens = extract_otsl_tokens(otsl_text)
    rows = []
    current_row = []
    for token in otsl_tokens:
        if token == "nl":
            if current_row:
                rows.append(current_row)
                current_row = []
            continue
        current_row.append(token)
    if current_row:
        rows.append(current_row)

    num_rows = len(rows)
    num_cols = max((len(row) for row in rows), default=0)
    grid = [row + [""] * (num_cols - len(row)) for row in rows]

    table_cells = []
    cell_tokens = {"fcel", "ecel", "ched", "rhed", "srow"}
    for row_idx, row in enumerate(grid):
        for col_idx, tag in enumerate(row):
            if tag not in cell_tokens:
                continue

            colspan = 1
            for col_scan in range(col_idx + 1, num_cols):
                if grid[row_idx][col_scan] == "lcel":
                    colspan += 1
                else:
                    break

            rowspan = 1
            for row_scan in range(row_idx + 1, num_rows):
                if grid[row_scan][col_idx] == "ucel":
                    rowspan += 1
                else:
                    break

            table_cells.append(
                TableCell(
                    text="",
                    row_span=rowspan,
                    col_span=colspan,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + rowspan,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + colspan,
                    column_header=tag == "ched",
                    row_header=tag == "rhed",
                    row_section=tag == "srow",
                )
            )

    return TableData(table_cells=table_cells, num_rows=num_rows, num_cols=num_cols)


def otsl_to_html(otsl_text: str) -> str:
    """Convert an OTSL string into Docling's HTML table serialization."""
    temp_doc = DoclingDocument(name="tableformer_v2_normalized")
    temp_table = temp_doc.add_table(data=build_table_data_from_otsl(otsl_text))
    return temp_table.export_to_html(temp_doc)


def normalize_v2_table_output(page_result: dict) -> dict:
    """Project native V2 output to a simpler model-agnostic table structure."""
    otsl_text = page_result["generated_otsl"]
    table_data = build_table_data_from_otsl(otsl_text)
    normalized_cells = []

    for idx, cell in enumerate(table_data.table_cells):
        bbox = None
        if idx < len(page_result["predicted_bboxes_crop_xyxy"]):
            x1, y1, x2, y2 = page_result["predicted_bboxes_crop_xyxy"][idx]
            bbox = BoundingBox(
                l=x1,
                t=y1,
                r=x2,
                b=y2,
                coord_origin=CoordOrigin.TOPLEFT,
            ).model_dump()

        normalized_cells.append(
            {
                "bbox": bbox,
                "text": cell.text,
                "row_span": cell.row_span,
                "col_span": cell.col_span,
                "start_row": cell.start_row_offset_idx,
                "end_row": cell.end_row_offset_idx,
                "start_col": cell.start_col_offset_idx,
                "end_col": cell.end_col_offset_idx,
                "column_header": cell.column_header,
                "row_header": cell.row_header,
                "row_section": cell.row_section,
            }
        )

    return {
        "table_bbox_page": page_result["table_bbox_page"],
        "num_rows": table_data.num_rows,
        "num_cols": table_data.num_cols,
        "cells": normalized_cells,
        "structure": {
            "otsl_text": otsl_text,
            "html": otsl_to_html(otsl_text),
        },
    }


def render_v2_table_visualization(
    image_path: Path,
    page_result: dict,
    output_path: Path,
) -> None:
    """Draw the predicted V2 table crop and its cell boxes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        x1, y1, x2, y2 = page_result["table_bbox_page"]
        draw.rectangle((x1, y1, x2, y2), outline="blue", width=4)

        for bbox in page_result.get("predicted_bboxes_crop_xyxy", []):
            bx1 = x1 + bbox[0]
            by1 = y1 + bbox[1]
            bx2 = x1 + bbox[2]
            by2 = y1 + bbox[3]
            draw.rectangle((bx1, by1, bx2, by2), outline="red", width=2)

        canvas.save(output_path)


def main() -> None:
    args = parse_args()

    if args.mode == "official":
        specs = OFFICIAL_SAMPLE_SPECS
    elif args.mode == "custom":
        specs = [build_custom_spec(args)]
    else:
        specs = build_hard_table_specs(args.dataset_json)

    model, tokenizer, artifact_ref = load_model_and_tokenizer(args)
    transform = make_transform(args.image_size)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    summary = []
    for spec in specs:
        page_results = []
        with Image.open(spec["image_path"]) as page_img:
            page_rgb = page_img.convert("RGB")

            for table_idx, table_bbox in enumerate(spec["table_bboxes"]):
                # V2 works directly on table crops, not on IOCR page payloads.
                x1, y1, x2, y2 = table_bbox
                table_crop = page_rgb.crop((x1, y1, x2, y2))
                crop_w, crop_h = table_crop.size
                image_tensor = transform(table_crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model.generate(
                        images=image_tensor,
                        tokenizer=tokenizer,
                        max_length=args.max_length,
                    )

                generated_ids = output["generated_ids"]
                decoded_otsl = tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                )

                pred_bboxes = output.get("predicted_bboxes")
                valid_bboxes = []
                crop_bboxes = []
                if pred_bboxes is not None and pred_bboxes.numel() > 0:
                    # The model pads bbox tensors; keep only non-zero entries.
                    pred_bboxes = pred_bboxes[0]
                    valid_mask = pred_bboxes.sum(dim=-1) > 0
                    pred_bboxes = pred_bboxes[valid_mask]
                    valid_bboxes = pred_bboxes.detach().cpu().tolist()

                    # Convert normalized [0, 1] coordinates back to crop pixel space for inspection.
                    for bbox in valid_bboxes:
                        crop_bboxes.append(
                            [
                                bbox[0] * crop_w,
                                bbox[1] * crop_h,
                                bbox[2] * crop_w,
                                bbox[3] * crop_h,
                            ]
                        )

                page_results.append(
                    {
                        "table_index": table_idx,
                        "table_bbox_page": table_bbox,
                        "crop_size": [crop_w, crop_h],
                        "generated_otsl": decoded_otsl,
                        "generated_ids": generated_ids[0].detach().cpu().tolist(),
                        "predicted_bboxes_normalized": valid_bboxes,
                        "predicted_bboxes_crop_xyxy": crop_bboxes,
                    }
                )

        viz_paths = []
        normalized_results = []
        for page_result in page_results:
            normalized_results.append(normalize_v2_table_output(page_result))
            viz_path = args.output_dir / "viz" / f"{spec['name']}.table_{page_result['table_index']}.png"
            render_v2_table_visualization(spec["image_path"], page_result, viz_path)
            viz_paths.append(str(viz_path))

        sample_output = {
            "sample": spec["name"],
            "model_version": "tableformer_v2",
            "artifact_ref": artifact_ref,
            "image_path": str(spec["image_path"]),
            "table_bboxes": spec["table_bboxes"],
            # Keep the native model output untouched for later fair comparison.
            "raw_output": page_results,
            # Keep a legacy alias for compatibility with older inspection scripts.
            "results": page_results,
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
                "tables_predicted": len(page_results),
                "debug_visualizations": viz_paths,
                "output_file": str(args.output_dir / f"{spec['name']}.pred.json"),
            }
        )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Direct TableFormer V2 inference finished.")
    print(f"Artifact: {artifact_ref}")
    print(f"Output dir: {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
