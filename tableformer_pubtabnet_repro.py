"""Run a public TableFormer V1 reproduction attempt on PubTabNet data.

This script uses the public `docling-ibm-models` TableFormer V1 predictor and
the public TFLOP PubTabNet bundle that is already available locally.

Why this script exists:
- the public TableFormer repo exposes inference, but not a ready-made benchmark
  reproduction path for PubTabNet
- the public PubTabNet metadata contains cell bboxes, cell tokens and the table
  structure sequence, which is enough to build a practical predictor input
- the official TFLOP bundle already includes PubTabNet metadata and images

What this script does:
1. read PubTabNet metadata from the public TFLOP bundle
2. create a page-like TableFormer input object with:
   - page image
   - table bbox
   - token list derived from PubTabNet cells
3. run `TFPredictor.multi_table_predict(...)`
4. normalize the native output and evaluate it with TEDS / TEDS-S

This is a strong public reproduction approximation for TableFormer on PubTabNet,
but it is still an adapter path, because the public repo does not ship an
official end-to-end PubTabNet benchmark script like TFLOP does.
"""

from __future__ import annotations

import argparse
import json
import re
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
PUBTABNET_JSONL = BASE_DIR / "data" / "TFLOP-dataset" / "meta_data" / "PubTabNet_2.0.0.jsonl"
PUBTABNET_TEST_GT = BASE_DIR / "data" / "TFLOP-dataset" / "meta_data" / "final_eval_v2.json"
PUBTABNET_IMAGES_DIR = BASE_DIR / "data" / "TFLOP-dataset" / "images"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "tableformer_pubtabnet_repro"

if str(IBM_MODELS_REPO) not in sys.path:
    sys.path.insert(0, str(IBM_MODELS_REPO))
if str(TFLOP_REPO) not in sys.path:
    sys.path.insert(0, str(TFLOP_REPO))

from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from tflop.evaluator import TEDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a public TableFormer V1 reproduction attempt on PubTabNet."
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help=(
            "PubTabNet split to evaluate. `val` is the strongest public path for "
            "TableFormer because the public JSONL carries cell annotations there. "
            "`test` keeps the official GT HTML, but the matching test cell "
            "annotations are not publicly present in the TFLOP bundle."
        ),
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=1000,
        help="Number of PubTabNet tables to evaluate.",
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
        "--offset",
        type=int,
        default=0,
        help="Start offset inside the selected PubTabNet split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "subset_1000",
        help="Directory for predictions, summaries and debug visualizations.",
    )
    parser.add_argument(
        "--pubtabnet-jsonl",
        type=Path,
        default=PUBTABNET_JSONL,
        help="PubTabNet JSONL annotation file.",
    )
    parser.add_argument(
        "--pubtabnet-images-dir",
        type=Path,
        default=PUBTABNET_IMAGES_DIR,
        help="Directory containing PubTabNet images or split subdirectories.",
    )
    parser.add_argument(
        "--pubtabnet-test-gt",
        type=Path,
        default=PUBTABNET_TEST_GT,
        help="Optional PubTabNet test GT JSON file.",
    )
    parser.add_argument(
        "--filenames-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with PubTabNet filenames or sample stems to "
            "evaluate. This is useful for fair A/B comparisons against an "
            "existing run on exactly the same samples."
        ),
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip writing debug visualizations.",
    )
    parser.add_argument(
        "--html-mode",
        choices=["cell_grid", "html_seq", "html_seq_cellmap"],
        default="cell_grid",
        help=(
            "How to build comparable HTML from the native V1 output. "
            "`cell_grid` rebuilds a logical grid from matched cells and currently "
            "gives the best overall TEDS in our public reproduction path. "
            "`html_seq` keeps the model's native HTML token stream and injects text. "
            "`html_seq_cellmap` also keeps the native HTML token stream, but maps "
            "matched text by predicted row/column to avoid shifts caused by empty cells."
        ),
    )
    parser.add_argument(
        "--token-mode",
        choices=["cell", "word"],
        default="cell",
        help=(
            "How to derive IOCR-like input tokens from PubTabNet cells. "
            "`cell` creates one token per GT cell. `word` splits cell text into "
            "smaller OCR-like word tokens and spreads them across the cell box."
        ),
    )
    parser.add_argument(
        "--cell-bbox-mode",
        choices=["original", "reconstructed"],
        default="original",
        help=(
            "How to handle PubTabNet cell bounding boxes before building the "
            "Docling/IOCR-like input. `original` keeps the public annotations. "
            "`reconstructed` fills missing cell bboxes from the finest-grained "
            "HTML grid, approximating the preprocessing step described in the "
            "TableFormer paper."
        ),
    )
    parser.add_argument(
        "--include-empty-cell-tokens",
        action="store_true",
        help=(
            "Experimental: include placeholder IOCR tokens for empty cells with "
            "reconstructed bboxes. This is not part of normal OCR, but is useful "
            "to test whether explicit empty-cell boxes affect Docling matching."
        ),
    )
    parser.add_argument("--min-rows", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--min-cols", type=int, default=None)
    parser.add_argument("--max-cols", type=int, default=None)
    return parser.parse_args()


def strip_xml_declarations(text: str) -> str:
    return text.replace("<?xml version='1.0' encoding='UTF-8'?>", "").strip()


def ensure_html_table_document(text: str) -> str:
    cleaned = strip_xml_declarations(text)
    lower = cleaned.lower()
    if "<html" in lower:
        return cleaned
    if "<table" in lower:
        return f"<html><body>{cleaned}</body></html>"
    return f"<html><body><table>{cleaned}</table></body></html>"


def canonicalize_pubtabnet_sections(text: str) -> str:
    """Wrap direct table rows after a thead into tbody, matching PubTabNet GT."""
    html_doc = ensure_html_table_document(text)
    return re.sub(
        r"(</thead>)((?:<tr>.*?</tr>)+)(</table>)",
        r"\1<tbody>\2</tbody>\3",
        html_doc,
        flags=re.DOTALL,
    )


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def visible_cell_text(text: str) -> str:
    """Return text content after removing inline HTML tags."""
    return normalize_whitespace(re.sub(r"<[^>]+>", "", text))


def table_shape_from_tokens(tokens: list[str]) -> tuple[int, int]:
    """Estimate logical table dimensions from PubTabNet/FinTabNet HTML tokens."""
    n_rows = 0
    max_cols = 0
    in_row = False
    col_count = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("<tr"):
            in_row = True
            col_count = 0
        elif token.startswith("</tr") and in_row:
            n_rows += 1
            max_cols = max(max_cols, col_count)
            in_row = False
        elif in_row and (token.startswith("<td") or token.startswith("<th")):
            attrs = token
            j = i
            if ">" not in token:
                j = i + 1
                while j < len(tokens):
                    attrs += tokens[j]
                    if ">" in tokens[j]:
                        break
                    j += 1
            colspan_match = re.search(r'colspan="(\d+)"', attrs)
            colspan = int(colspan_match.group(1)) if colspan_match else 1
            col_count += max(colspan, 1)
            i = j
        i += 1
    return n_rows, max_cols


def row_matches_shape_filter(row: dict[str, Any], args: argparse.Namespace) -> bool:
    n_rows, n_cols = table_shape_from_tokens(row["html"]["structure"]["tokens"])
    checks = [
        args.min_rows is None or n_rows >= args.min_rows,
        args.max_rows is None or n_rows <= args.max_rows,
        args.min_cols is None or n_cols >= args.min_cols,
        args.max_cols is None or n_cols <= args.max_cols,
    ]
    return all(checks)


def load_requested_filenames(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    requested: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        requested.add(value)
        requested.add(Path(value).stem)
        requested.add(f"{Path(value).stem}.png")
    return requested


def load_predictor_config() -> dict[str, Any]:
    config_path = LOCAL_TF_ACCURATE_DIR / "tm_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["save_dir"] = str(LOCAL_TF_ACCURATE_DIR)
    return config


def load_pubtabnet_test_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def image_dir_for_split(split: str, images_root: Path = PUBTABNET_IMAGES_DIR) -> Path:
    if (images_root / split).exists():
        return images_root / split
    if split == "val":
        if (images_root / "validation").exists():
            return images_root / "validation"
    if (images_root / "test").exists():
        return images_root / "test"
    return images_root


def load_pubtabnet_rows(
    split: str,
    jsonl_path: Path = PUBTABNET_JSONL,
    images_root: Path = PUBTABNET_IMAGES_DIR,
    test_gt_path: Path = PUBTABNET_TEST_GT,
) -> list[dict[str, Any]]:
    gt = load_pubtabnet_test_ground_truth(test_gt_path) if split == "test" else {}
    image_dir = image_dir_for_split(split, images_root)
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            filename = row.get("filename")
            if split == "val":
                if row.get("split") != "val":
                    continue
            else:
                # The public TFLOP bundle ships final_eval_v2.json for the
                # official test GT HTML, but not the matching PubTabNet test
                # rows with cell bboxes in PubTabNet_2.0.0.jsonl.
                continue
            if split == "test" and filename not in gt:
                continue
            image_path = image_dir / filename
            if not image_path.exists():
                continue
            rows.append(row)
    return rows


def rich_text_from_tokens(tokens: list[str]) -> str:
    return "".join(tokens or [])


def pubtabnet_html_to_document(row: dict[str, Any], gt_by_filename: dict[str, dict[str, Any]]) -> str:
    """Prefer the official final_eval HTML when available."""
    filename = row["filename"]
    official = gt_by_filename.get(filename, {}).get("html")
    if official:
        return ensure_html_table_document(official)

    cells = row["html"]["cells"]
    cell_iter = iter(cells)
    parts: list[str] = ["<html><body><table>"]
    for token in row["html"]["structure"]["tokens"]:
        if token == "</td>":
            cell = next(cell_iter)
            parts.append(rich_text_from_tokens(cell.get("tokens", [])))
            parts.append(token)
        else:
            parts.append(token)
    parts.append("</table></body></html>")
    return ensure_html_table_document("".join(parts))


def cell_bbox_xyxy(cell: dict[str, Any]) -> list[float]:
    bbox = cell.get("bbox", [])
    if len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


def span_from_attrs(attrs: str, name: str) -> int:
    match = re.search(rf'{name}="(\d+)"', attrs)
    return int(match.group(1)) if match else 1


def is_table_cell_open_token(token: str) -> bool:
    """Return True for real <td>/<th> openings, but not <thead>/<tbody>."""
    return bool(re.match(r"^<(td|th)(\s|>|$)", token))


def parse_pubtabnet_grid_cells(row: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    """Map PubTabNet cells onto the finest-grained logical table grid.

    PubTabNet stores table structure as HTML-like tokens and cell content in a
    separate ``html.cells`` list. The cells list is aligned with the order of
    ``<td>...</td>`` elements in the structure tokens. This parser reconstructs
    each cell's logical row/column offsets while respecting rowspan/colspan.
    """
    tokens = row["html"]["structure"]["tokens"]
    cells = row["html"]["cells"]
    records: list[dict[str, Any]] = []
    occupied: set[tuple[int, int]] = set()
    row_idx = -1
    col_idx = 0
    max_cols = 0
    cell_idx = 0
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if token.startswith("<tr"):
            row_idx += 1
            col_idx = 0
            i += 1
            continue

        if is_table_cell_open_token(token):
            attrs = token
            while ">" not in attrs and i + 1 < len(tokens):
                i += 1
                attrs += tokens[i]
            while (row_idx, col_idx) in occupied:
                col_idx += 1

            row_span = max(1, span_from_attrs(attrs, "rowspan"))
            col_span = max(1, span_from_attrs(attrs, "colspan"))
            if cell_idx >= len(cells):
                break
            cell = cells[cell_idx]
            records.append(
                {
                    "cell_index": cell_idx,
                    "cell": cell,
                    "start_row": row_idx,
                    "end_row": row_idx + row_span,
                    "start_col": col_idx,
                    "end_col": col_idx + col_span,
                    "row_span": row_span,
                    "col_span": col_span,
                }
            )
            for span_row in range(row_idx + 1, row_idx + row_span):
                for span_col in range(col_idx, col_idx + col_span):
                    occupied.add((span_row, span_col))
            col_idx += col_span
            max_cols = max(max_cols, col_idx)
            cell_idx += 1
        i += 1

    return records, row_idx + 1, max_cols


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def fill_grid_lines(
    candidates: list[list[float]],
    fallback_min: float,
    fallback_max: float,
) -> list[float]:
    """Convert sparse boundary candidates into monotonic grid lines."""
    n = len(candidates) - 1
    known: list[float | None] = [median_or_none(values) for values in candidates]

    # Prefer observed outer table boundaries when available; otherwise use the
    # image/table extent fallback.
    if known[0] is None:
        known[0] = fallback_min
    if known[n] is None:
        known[n] = fallback_max

    for idx, value in enumerate(known):
        if value is not None:
            continue
        left = idx - 1
        while left >= 0 and known[left] is None:
            left -= 1
        right = idx + 1
        while right <= n and known[right] is None:
            right += 1
        if left >= 0 and right <= n and known[left] is not None and known[right] is not None:
            ratio = (idx - left) / (right - left)
            known[idx] = float(known[left]) + ratio * (float(known[right]) - float(known[left]))
        elif left >= 0 and known[left] is not None:
            known[idx] = float(known[left])
        elif right <= n and known[right] is not None:
            known[idx] = float(known[right])
        else:
            known[idx] = fallback_min + (fallback_max - fallback_min) * idx / max(1, n)

    lines = [float(value) for value in known]
    for idx in range(1, len(lines)):
        if lines[idx] < lines[idx - 1]:
            lines[idx] = lines[idx - 1]
    return lines


def reconstruct_missing_cell_bboxes(
    row: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill missing PubTabNet cell bboxes from the logical grid.

    This approximates the TableFormer paper's preprocessing idea: parse the
    table structure, build the finest-grained grid, and assign missing cell
    boxes from row/column grid boundaries inferred from existing cell boxes.
    """
    prepared = deepcopy(row)
    records, num_rows, num_cols = parse_pubtabnet_grid_cells(prepared)
    x_candidates = [[] for _ in range(num_cols + 1)]
    y_candidates = [[] for _ in range(num_rows + 1)]
    observed_x0: list[float] = []
    observed_y0: list[float] = []
    observed_x1: list[float] = []
    observed_y1: list[float] = []

    for record in records:
        bbox = record["cell"].get("bbox", [])
        if len(bbox) < 4:
            continue
        x0, y0, x1, y1 = cell_bbox_xyxy(record["cell"])
        x_candidates[record["start_col"]].append(x0)
        x_candidates[record["end_col"]].append(x1)
        y_candidates[record["start_row"]].append(y0)
        y_candidates[record["end_row"]].append(y1)
        observed_x0.append(x0)
        observed_y0.append(y0)
        observed_x1.append(x1)
        observed_y1.append(y1)

    fallback_x0 = min(observed_x0) if observed_x0 else 0.0
    fallback_y0 = min(observed_y0) if observed_y0 else 0.0
    fallback_x1 = max(observed_x1) if observed_x1 else float(image_width)
    fallback_y1 = max(observed_y1) if observed_y1 else float(image_height)
    x_lines = fill_grid_lines(x_candidates, fallback_x0, fallback_x1)
    y_lines = fill_grid_lines(y_candidates, fallback_y0, fallback_y1)

    reconstructed = 0
    empty_reconstructed = 0
    for record in records:
        cell = record["cell"]
        text = rich_text_from_tokens(cell.get("tokens", []))
        is_empty = not normalize_whitespace(re.sub(r"<[^>]+>", "", text))
        cell["is_empty"] = is_empty
        if len(cell.get("bbox", [])) >= 4:
            cell["bbox_reconstructed"] = False
            continue
        x0 = x_lines[record["start_col"]]
        x1 = x_lines[record["end_col"]]
        y0 = y_lines[record["start_row"]]
        y1 = y_lines[record["end_row"]]
        cell["bbox"] = [
            max(0.0, min(float(image_width), x0)),
            max(0.0, min(float(image_height), y0)),
            max(0.0, min(float(image_width), x1)),
            max(0.0, min(float(image_height), y1)),
        ]
        cell["bbox_reconstructed"] = True
        reconstructed += 1
        if is_empty:
            empty_reconstructed += 1

    stats = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "total_cells": len(records),
        "reconstructed_bboxes": reconstructed,
        "reconstructed_empty_bboxes": empty_reconstructed,
    }
    return prepared, stats


def table_bbox_from_cells(cells: list[dict[str, Any]], image_width: int, image_height: int) -> list[float]:
    if not cells:
        return [0.0, 0.0, float(image_width), float(image_height)]
    xs0, ys0, xs1, ys1 = [], [], [], []
    for cell in cells:
        x0, y0, x1, y1 = cell_bbox_xyxy(cell)
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)
    return [
        max(0.0, min(xs0)),
        max(0.0, min(ys0)),
        min(float(image_width), max(xs1)),
        min(float(image_height), max(ys1)),
    ]


def split_text_like_words(text: str) -> list[str]:
    """Split rich PubTabNet cell text into small OCR-like chunks."""
    stripped = text.strip()
    if not stripped:
        return []
    return stripped.split()


def boxes_for_word_splits(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    parts: list[str],
) -> list[list[float]]:
    """Create simple left-to-right sub-boxes inside one GT cell box."""
    if not parts:
        return []
    width = max(1.0, x1 - x0)
    total = sum(max(1, len(part)) for part in parts)
    cursor = x0
    boxes: list[list[float]] = []
    for idx, part in enumerate(parts):
        weight = max(1, len(part)) / total
        next_x = x1 if idx == len(parts) - 1 else cursor + width * weight
        boxes.append([cursor, y0, max(cursor, next_x), y1])
        cursor = next_x
    return boxes


def build_iocr_tokens_from_cells(
    cells: list[dict[str, Any]],
    token_mode: str,
    include_empty_cell_tokens: bool = False,
) -> list[dict[str, Any]]:
    """Build a minimal IOCR token list from PubTabNet cell annotations.

    We use one text token per GT cell. This is not a full PDF-word IOCR export,
    but it gives TableFormer a page token inventory for cell matching and keeps
    the public reproduction path simple and deterministic.
    """
    tokens: list[dict[str, Any]] = []
    token_id = 1
    for idx, cell in enumerate(cells):
        if len(cell.get("bbox", [])) < 4:
            continue
        x0, y0, x1, y1 = cell_bbox_xyxy(cell)
        text = rich_text_from_tokens(cell.get("tokens", []))
        if not visible_cell_text(text):
            if not include_empty_cell_tokens:
                continue
            text = " "

        if token_mode == "word":
            parts = split_text_like_words(text)
            sub_boxes = boxes_for_word_splits(x0, y0, x1, y1, parts)
            for part_idx, (part, box) in enumerate(zip(parts, sub_boxes)):
                bx0, by0, bx1, by1 = box
                tokens.append(
                    {
                        "id": token_id,
                        "text": part,
                        "bbox": {"l": bx0, "t": by0, "r": bx1, "b": by1},
                        "block_id": 0,
                        "text_line_id": idx,
                        "indexInLine": part_idx,
                        "confidence": 1.0,
                        "word_in_presentation_ltr_order": part,
                        "lang": "en",
                    }
                )
                token_id += 1
            continue

        tokens.append(
            {
                "id": token_id,
                "text": text,
                "bbox": {"l": x0, "t": y0, "r": x1, "b": y1},
                "block_id": 0,
                "text_line_id": idx,
                "indexInLine": 0,
                "confidence": 1.0,
                "word_in_presentation_ltr_order": text,
                "lang": "en",
            }
        )
        token_id += 1
    return tokens


def load_page_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"))


def prepare_pubtabnet_page(
    row: dict[str, Any],
    image_path: Path,
    token_mode: str,
    cell_bbox_mode: str,
    include_empty_cell_tokens: bool,
) -> tuple[dict[str, Any], list[float], dict[str, Any]]:
    image = load_page_image(image_path)
    height, width = image.shape[:2]
    reconstruction_stats: dict[str, Any] = {
        "cell_bbox_mode": cell_bbox_mode,
        "reconstructed_bboxes": 0,
        "reconstructed_empty_bboxes": 0,
    }
    if cell_bbox_mode == "reconstructed":
        row, reconstruction_stats = reconstruct_missing_cell_bboxes(row, width, height)
        reconstruction_stats["cell_bbox_mode"] = cell_bbox_mode
    cells = row["html"]["cells"]
    table_bbox = table_bbox_from_cells(cells, width, height)
    page = {
        "image": image,
        "png_image_fn": str(image_path),
        "tokens": build_iocr_tokens_from_cells(
            cells,
            token_mode,
            include_empty_cell_tokens=include_empty_cell_tokens,
        ),
        "width": width,
        "height": height,
        "table_bboxes": [table_bbox],
    }
    return page, table_bbox, reconstruction_stats


def to_v1_otsl_text(rs_seq: list[str]) -> str:
    return " ".join(f"<{token}>" for token in rs_seq)


def to_v1_html_text(html_seq: list[str]) -> str:
    return f"<table>{''.join(html_seq)}</table>"


def text_from_v1_cell(cell: dict[str, Any]) -> str:
    """Extract matched text from the native V1 response cell."""
    text_boxes = cell.get("text_cell_bboxes") or []
    parts = [str(box.get("token", "")) for box in text_boxes if box.get("token")]
    return "".join(parts)


def html_from_normalized_cells(cells: list[dict[str, Any]], num_rows: int | None, num_cols: int | None) -> str:
    """Render a simple HTML table from logical TableFormer cells."""
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


def html_from_v1_sequence_with_text(html_seq: list[str], tf_responses: list[dict[str, Any]]) -> str:
    """Inject matched cell text into the native V1 HTML token stream.

    This keeps the model's predicted structure intact instead of rebuilding the
    table grid from logical cells, which can lose details such as sectioning or
    span placement.
    """
    if not html_seq:
        return ""

    cell_iter = iter(tf_responses)
    parts: list[str] = ["<table>"]
    for token in html_seq:
        if token == "</td>":
            cell = next(cell_iter, None)
            if cell is not None:
                parts.append(text_from_v1_cell(cell))
            parts.append(token)
        else:
            parts.append(token)
    parts.append("</table>")
    return "".join(parts)


def span_from_tag(tag: str, name: str) -> int:
    match = re.search(rf'{name}="(\d+)"', tag)
    return int(match.group(1)) if match else 1


def html_from_v1_sequence_with_cell_map(
    html_seq: list[str],
    tf_responses: list[dict[str, Any]],
) -> str:
    """Inject matched text into native V1 HTML by predicted grid coordinates.

    ``html_seq`` contains empty cells while ``tf_responses`` usually does not.
    Sequential text injection can therefore shift text into the wrong cells.
    Mapping by ``start_row_offset_idx`` / ``start_col_offset_idx`` keeps empty
    cells empty and preserves the model's native HTML structure.
    """
    if not html_seq:
        return ""

    cell_text_by_start = {
        (int(cell["start_row_offset_idx"]), int(cell["start_col_offset_idx"])): text_from_v1_cell(cell)
        for cell in tf_responses
    }

    parts: list[str] = ["<table>"]
    row_idx = -1
    col_idx = 0
    occupied: set[tuple[int, int]] = set()
    open_cell: dict[str, int] | None = None
    pending_tag: list[str] = []

    def advance_to_free_col() -> None:
        nonlocal col_idx
        while (row_idx, col_idx) in occupied:
            col_idx += 1

    for token in html_seq:
        if pending_tag:
            pending_tag.append(token)
            parts.append(token)
            if token == ">":
                tag = "".join(pending_tag)
                if tag.startswith("<td") or tag.startswith("<th"):
                    advance_to_free_col()
                    open_cell = {
                        "row": row_idx,
                        "col": col_idx,
                        "rowspan": span_from_tag(tag, "rowspan"),
                        "colspan": span_from_tag(tag, "colspan"),
                    }
                pending_tag = []
            continue

        if token == "<tr>":
            row_idx += 1
            col_idx = 0
            parts.append(token)
            continue

        if token in {"<td", "<th"}:
            pending_tag = [token]
            parts.append(token)
            continue

        if token.startswith("<td") or token.startswith("<th"):
            parts.append(token)
            advance_to_free_col()
            open_cell = {
                "row": row_idx,
                "col": col_idx,
                "rowspan": span_from_tag(token, "rowspan"),
                "colspan": span_from_tag(token, "colspan"),
            }
            continue

        if token in {"</td>", "</th>"} and open_cell is not None:
            parts.append(cell_text_by_start.get((open_cell["row"], open_cell["col"]), ""))
            parts.append(token)
            for span_row in range(open_cell["row"] + 1, open_cell["row"] + open_cell["rowspan"]):
                for span_col in range(open_cell["col"], open_cell["col"] + open_cell["colspan"]):
                    occupied.add((span_row, span_col))
            col_idx = open_cell["col"] + open_cell["colspan"]
            open_cell = None
            continue

        parts.append(token)

    parts.append("</table>")
    return "".join(parts)


def normalize_v1_table_output(
    tf_output: dict[str, Any],
    table_bbox: list[float],
    html_mode: str,
) -> dict[str, Any]:
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
    if html_mode == "html_seq":
        rendered_html = html_from_v1_sequence_with_text(html_seq, tf_responses)
    elif html_mode == "html_seq_cellmap":
        rendered_html = canonicalize_pubtabnet_sections(
            html_from_v1_sequence_with_cell_map(html_seq, tf_responses)
        )
    else:
        rendered_html = html_from_normalized_cells(normalized_cells, num_rows, num_cols)

    return {
        "table_bbox_page": table_bbox,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "cells": normalized_cells,
        "structure": {
            "otsl_tokens": rs_seq,
            "otsl_text": to_v1_otsl_text(rs_seq),
            "html": rendered_html or (to_v1_html_text(html_seq) if html_seq else ""),
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


def teds_scores(pred_html: str, gt_html: str) -> tuple[float, float]:
    metric_full = TEDS(structure_only=False)
    metric_struct = TEDS(structure_only=True)
    return (
        float(metric_full.evaluate(pred_html, gt_html)),
        float(metric_struct.evaluate(pred_html, gt_html)),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(payload), indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    gt_by_filename = load_pubtabnet_test_ground_truth(args.pubtabnet_test_gt)
    all_rows = load_pubtabnet_rows(
        args.split,
        args.pubtabnet_jsonl,
        args.pubtabnet_images_dir,
        args.pubtabnet_test_gt,
    )
    all_rows = [row for row in all_rows if row_matches_shape_filter(row, args)]
    requested_filenames = load_requested_filenames(args.filenames_file)
    if requested_filenames is not None:
        all_rows = [
            row
            for row in all_rows
            if row.get("filename") in requested_filenames
            or Path(row.get("filename", "")).stem in requested_filenames
        ]
    if not all_rows:
        raise RuntimeError(
            f"No public PubTabNet rows found for split={args.split}. "
            "The public TFLOP bundle exposes cell-annotated rows in "
            "PubTabNet_2.0.0.jsonl for train/val, but not for the official test split."
        )
    subset = all_rows[args.offset : args.offset + args.subset_size]

    config = load_predictor_config()
    predictor = TFPredictor(config, device=args.device, num_threads=args.num_threads)

    per_table: list[dict[str, Any]] = []
    teds_values: list[float] = []
    teds_s_values: list[float] = []

    for row_idx, row in enumerate(subset, start=args.offset):
        filename = row["filename"]
        image_path = image_dir_for_split(args.split, args.pubtabnet_images_dir) / filename
        gt_html = pubtabnet_html_to_document(row, gt_by_filename)
        iocr_page, table_bbox, reconstruction_stats = prepare_pubtabnet_page(
            row,
            image_path,
            args.token_mode,
            args.cell_bbox_mode,
            args.include_empty_cell_tokens,
        )

        raw_results = predictor.multi_table_predict(
            iocr_page,
            deepcopy([table_bbox]),
            do_matching=True,
            correct_overlapping_cells=False,
            sort_row_col_indexes=True,
        )
        raw_output = raw_results[0]
        normalized_output = normalize_v1_table_output(raw_output, table_bbox, args.html_mode)
        pred_html = ensure_html_table_document(
            normalized_output["structure"].get("html") or ""
        )
        teds, teds_s = teds_scores(pred_html, gt_html)
        teds_values.append(teds)
        teds_s_values.append(teds_s)

        viz_path = None
        if not args.no_viz:
            viz_path = args.output_dir / "viz" / f"{Path(filename).stem}.png"
            render_v1_table_visualization(image_path, table_bbox, raw_output, viz_path)

        sample_payload = {
            "sample": Path(filename).stem,
            "dataset": "PubTabNet",
            "split": args.split,
            "sample_index": row_idx,
            "image_path": str(image_path),
            "model_version": "tableformer_v1",
            "weights_dir": str(LOCAL_TF_ACCURATE_DIR),
            "table_bboxes": [table_bbox],
            "cell_bbox_preprocessing": reconstruction_stats,
            "raw_output": raw_results,
            "normalized_output": [normalized_output],
            "ground_truth": {
                "gt_html": gt_html,
                "type": gt_by_filename.get(filename, {}).get("type"),
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
                "type": gt_by_filename.get(filename, {}).get("type"),
                "teds": teds,
                "teds_s": teds_s,
                "pred_path": str(args.output_dir / f"{Path(filename).stem}.pred.json"),
                "cell_bbox_preprocessing": reconstruction_stats,
            }
        )

    summary = {
        "dataset": "PubTabNet",
        "split": args.split,
        "mode": "public_reproduction_attempt",
        "subset_size": len(subset),
        "offset": args.offset,
        "device": args.device,
        "html_mode": args.html_mode,
        "token_mode": args.token_mode,
        "cell_bbox_mode": args.cell_bbox_mode,
        "include_empty_cell_tokens": args.include_empty_cell_tokens,
        "filenames_file": None if args.filenames_file is None else str(args.filenames_file),
        "weights_dir": str(LOCAL_TF_ACCURATE_DIR),
        "source_repo": str(IBM_MODELS_REPO),
        "source_pubtabnet_jsonl": str(args.pubtabnet_jsonl),
        "source_pubtabnet_test_gt": str(args.pubtabnet_test_gt),
        "source_pubtabnet_images": str(image_dir_for_split(args.split, args.pubtabnet_images_dir)),
        "mean_teds": statistics.mean(teds_values) if teds_values else None,
        "mean_teds_s": statistics.mean(teds_s_values) if teds_s_values else None,
        "paper_reference": {
            "pubtabnet_all_teds": 0.9675,
            "pubtabnet_simple_teds": 0.9850,
            "pubtabnet_complex_teds": 0.9500,
        },
        "notes": [
            "This uses the public docling-ibm-models TableFormer V1 weights and predictor.",
            "The public repo does not ship an official PubTabNet benchmark adapter, so this script derives a minimal IOCR-like token list from PubTabNet cell annotations.",
            "The public TFLOP bundle appears to expose official test GT HTML but not the matching test cell annotation rows needed by TableFormer V1.",
            "When cell_bbox_mode=reconstructed, missing PubTabNet cell boxes are inferred from the GT HTML grid and existing cell boxes, approximating the TableFormer paper's prepared-format preprocessing.",
        ],
    }

    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "per_table.json", per_table)

    print("TableFormer PubTabNet reproduction attempt finished.")
    print(f"Subset size: {len(subset)}")
    print(f"Mean TEDS: {summary['mean_teds']}")
    print(f"Mean TEDS-S: {summary['mean_teds_s']}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
