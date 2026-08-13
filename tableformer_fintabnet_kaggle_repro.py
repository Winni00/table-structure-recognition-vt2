"""Run a small TableFormer V1 adaptation on Kaggle FinTabNet v1.0.0 PDFs.

This is a paper-nearer FinTabNet source than the OTSL-derived subset, but it is
still an adapter path: we render PDFs locally and construct IOCR-like tokens
from FinTabNet cell annotations.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
IBM_MODELS_REPO = BASE_DIR / "repo" / "docling-ibm-models"
TFLOP_REPO = BASE_DIR / "repo" / "TFLOP"
LOCAL_TF_ACCURATE_DIR = (
    BASE_DIR / "models" / "docling_ibm" / "tableformer" / "accurate"
)
KAGGLE_ROOT = BASE_DIR / "data" / "fintabnet_kaggle"
DEFAULT_TABLE_JSONL = KAGGLE_ROOT / "FinTabNet_1.0.0_table_val.jsonl"
DEFAULT_PDF_DIR = KAGGLE_ROOT / "pdfs"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "tableformer_fintabnet_kaggle" / "smoke_5"

if str(IBM_MODELS_REPO) not in sys.path:
    sys.path.insert(0, str(IBM_MODELS_REPO))
if str(TFLOP_REPO) not in sys.path:
    sys.path.insert(0, str(TFLOP_REPO))

from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from tflop.evaluator import TEDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-jsonl", type=Path, default=DEFAULT_TABLE_JSONL)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--render-scale", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument(
        "--html-mode",
        choices=["cell_grid", "html_seq", "html_seq_cellmap"],
        default="cell_grid",
    )
    parser.add_argument(
        "--token-source",
        choices=["cells", "pdftext", "tesseract"],
        default="cells",
        help="Use cell tokens from annotation or extract tokens from PDF text.",
    )
    parser.add_argument(
        "--only-filenames",
        type=Path,
        default=None,
        help="Optional text file with one PDF filename per line to restrict the run.",
    )
    parser.add_argument("--min-rows", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--min-cols", type=int, default=None)
    parser.add_argument("--max-cols", type=int, default=None)
    parser.add_argument("--save-debug-html", action="store_true")
    return parser.parse_args()


def load_predictor_config() -> dict[str, Any]:
    config_path = LOCAL_TF_ACCURATE_DIR / "tm_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["save_dir"] = str(LOCAL_TF_ACCURATE_DIR)
    return config


def ensure_html_table_document(text: str) -> str:
    cleaned = text.strip()
    if "<html" in cleaned.lower():
        return cleaned
    if "<table" in cleaned.lower():
        return f"<html><body>{cleaned}</body></html>"
    return f"<html><body><table>{cleaned}</table></body></html>"


def canonicalize_pubtabnet_sections(text: str) -> str:
    """Wrap direct table rows after a thead into tbody for TEDS HTML parity."""
    html_doc = ensure_html_table_document(text)
    return re.sub(
        r"(</thead>)((?:<tr>.*?</tr>)+)(</table>)",
        r"\1<tbody>\2</tbody>\3",
        html_doc,
        flags=re.DOTALL,
    )


def rich_text(tokens: list[str]) -> str:
    return "".join(tokens or [])


def table_shape_from_tokens(tokens: list[str]) -> tuple[int, int]:
    """Estimate logical table dimensions from tokenized HTML structure."""
    n_rows = 0
    max_cols = 0
    in_row = False
    col_count = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "<tr>":
            in_row = True
            col_count = 0
        elif token == "</tr>" and in_row:
            n_rows += 1
            max_cols = max(max_cols, col_count)
            in_row = False
        elif in_row and token.startswith("<td"):
            attrs = token
            j = i + 1
            while j < len(tokens) and tokens[j] != ">":
                attrs += tokens[j]
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


def html_from_fintabnet(row: dict[str, Any]) -> str:
    cells = row["html"]["cells"]
    cell_iter = iter(cells)
    parts: list[str] = []
    for token in row["html"]["structure"]["tokens"]:
        if token == "</td>":
            cell = next(cell_iter, {})
            parts.append(rich_text(cell.get("tokens", [])))
            parts.append(token)
        else:
            parts.append(token)
    return ensure_html_table_document("".join(parts))


def scale_bbox(
    bbox: list[float],
    scale: float,
    page_height: float | None = None,
    origin_y: float = 0.0,
) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    if page_height is None:
        return [x0 * scale, (y0 - origin_y) * scale, x1 * scale, (y1 - origin_y) * scale]

    # PDF coordinates are bottom-left based. Normalize against the origin,
    # then flip into image coordinates.
    return [
        x0 * scale,
        (page_height - (y1 - origin_y)) * scale,
        x1 * scale,
        (page_height - (y0 - origin_y)) * scale,
    ]


def clip_bbox(
    bbox: list[float],
    image_width: float,
    image_height: float,
) -> list[float]:
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(x0, image_width))
    x1 = max(0.0, min(x1, image_width))
    y0 = max(0.0, min(y0, image_height))
    y1 = max(0.0, min(y1, image_height))
    return [x0, y0, x1, y1]


def clamp_bbox_pdf(
    bbox: list[float],
    pdf_bounds: tuple[float, float, float, float],
) -> list[float]:
    x0, y0, x1, y1 = bbox
    pdf_x0, pdf_y0, pdf_x1, pdf_y1 = pdf_bounds
    x0 = max(pdf_x0, min(x0, pdf_x1))
    x1 = max(pdf_x0, min(x1, pdf_x1))
    y0 = max(pdf_y0, min(y0, pdf_y1))
    y1 = max(pdf_y0, min(y1, pdf_y1))
    return [x0, y0, x1, y1]


def bbox_is_valid(bbox: list[float]) -> bool:
    x0, y0, x1, y1 = bbox
    return (x1 - x0) > 1.0 and (y1 - y0) > 1.0


def table_bbox_from_cells(row: dict[str, Any]) -> list[float] | None:
    cells = row.get("html", {}).get("cells", [])
    coords: list[float] = []
    for cell in cells:
        bbox = cell.get("bbox")
        if not bbox:
            continue
        coords.extend([float(v) for v in bbox[:4]])
    if not coords:
        return None
    xs = coords[0::4]
    ys = coords[1::4]
    xes = coords[2::4]
    yes = coords[3::4]
    return [min(xs), min(ys), max(xes), max(yes)]


def scale_bbox_with_fallback(
    bbox: list[float],
    scale: float,
    page_width: float,
    page_height: float,
    origin_y: float = 0.0,
    pdf_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[list[float], str]:
    """Scale bbox from PDF coords to image coords with a safe fallback.

    Most FinTabNet bboxes are PDF bottom-left based, so we flip y. A small
    subset appears already top-left based. If the flipped bbox ends up empty,
    retry without flipping and pick the first valid result.
    """
    image_width = page_width * scale
    image_height = page_height * scale

    def try_scale(raw: list[float], tag: str) -> tuple[list[float], str] | None:
        flipped = scale_bbox(raw, scale, page_height, origin_y)
        flipped = clip_bbox(flipped, image_width, image_height)
        if bbox_is_valid(flipped):
            return flipped, f"{tag}_flip"

        unflipped = scale_bbox(raw, scale, None, origin_y)
        unflipped = clip_bbox(unflipped, image_width, image_height)
        if bbox_is_valid(unflipped):
            return unflipped, f"{tag}_noflip"
        return None

    raw_bbox = bbox
    if pdf_bounds is not None:
        raw_bbox = clamp_bbox_pdf(raw_bbox, pdf_bounds)
        candidate = try_scale(raw_bbox, "clamp")
        if candidate is not None:
            return candidate

    candidate = try_scale(bbox, "raw")
    if candidate is not None:
        return candidate

    return clip_bbox(scale_bbox(bbox, scale, page_height, origin_y), image_width, image_height), "invalid"


def render_pdf_page(
    page: "pdfium.PdfPage",
    scale: float,
    override_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[
    np.ndarray | None,
    tuple[float, float] | None,
    tuple[float, float, float, float] | None,
    tuple[float, float, float, float] | None,
]:
    mediabox = page.get_mediabox()
    cropbox = page.get_cropbox()

    # Render using the override bounds (if any) to align with FinTabNet coords.
    render_bounds = override_bounds or mediabox
    page.set_cropbox(*render_bounds)
    image = page.render(scale=scale).to_pil().convert("RGB")
    page.set_cropbox(*cropbox)

    arr = np.array(image)
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None, None, None, None

    page_size = (render_bounds[2] - render_bounds[0], render_bounds[3] - render_bounds[1])
    return arr, page_size, mediabox, cropbox


def union_pdf_bounds(
    bounds_list: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    x0 = min(b[0] for b in bounds_list)
    y0 = min(b[1] for b in bounds_list)
    x1 = max(b[2] for b in bounds_list)
    y1 = max(b[3] for b in bounds_list)
    return (x0, y0, x1, y1)


def get_pdf_boxes(
    page: "pdfium.PdfPage",
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    return page.get_mediabox(), page.get_cropbox()


def build_pdf_text_tokens(
    textpage: "pdfium.PdfTextPage",
    scale: float,
    page_width: float,
    page_height: float,
    origin_y: float,
    render_bounds: tuple[float, float, float, float],
    table_bbox_img: list[float] | None,
    text_offset: tuple[float, float] = (0.0, 0.0),
) -> list[dict[str, Any]]:
    total_chars = textpage.count_chars()
    tokens: list[dict[str, Any]] = []
    current_text: list[str] = []
    current_bbox: list[float] | None = None
    pdf_x0, pdf_y0, pdf_x1, pdf_y1 = render_bounds

    def flush_token() -> None:
        nonlocal current_text, current_bbox, tokens
        if not current_text or current_bbox is None:
            current_text = []
            current_bbox = None
            return
        x0, y0, x1, y1 = current_bbox
        bbox = [x0, y0, x1, y1]
        bbox = clamp_bbox_pdf(bbox, render_bounds)
        x0i, y0i, x1i, y1i = scale_bbox(bbox, scale, page_height, origin_y)
        clipped = clip_bbox([x0i, y0i, x1i, y1i], page_width * scale, page_height * scale)
        if not bbox_is_valid(clipped):
            current_text = []
            current_bbox = None
            return
        if table_bbox_img is not None:
            cx = (clipped[0] + clipped[2]) / 2
            cy = (clipped[1] + clipped[3]) / 2
            tbx0, tby0, tbx1, tby1 = table_bbox_img
            if not (tbx0 <= cx <= tbx1 and tby0 <= cy <= tby1):
                current_text = []
                current_bbox = None
                return
        text = "".join(current_text)
        tokens.append(
            {
                "id": len(tokens) + 1,
                "text": text,
                "bbox": {"l": clipped[0], "t": clipped[1], "r": clipped[2], "b": clipped[3]},
                "block_id": 0,
                "text_line_id": len(tokens),
                "indexInLine": 0,
                "confidence": 1.0,
                "word_in_presentation_ltr_order": text,
                "lang": "en",
            }
        )
        current_text = []
        current_bbox = None

    for idx in range(total_chars):
        ch = textpage.get_text_range(idx, 1) or ""
        if ch.isspace():
            flush_token()
            continue
        x0, y0, x1, y1 = textpage.get_charbox(idx)
        dx, dy = text_offset
        x0 += dx
        x1 += dx
        y0 += dy
        y1 += dy
        # Clamp to render bounds to avoid invalid coords
        x0 = max(pdf_x0, min(x0, pdf_x1))
        x1 = max(pdf_x0, min(x1, pdf_x1))
        y0 = max(pdf_y0, min(y0, pdf_y1))
        y1 = max(pdf_y0, min(y1, pdf_y1))
        if current_bbox is None:
            current_bbox = [x0, y0, x1, y1]
        else:
            current_bbox = [
                min(current_bbox[0], x0),
                min(current_bbox[1], y0),
                max(current_bbox[2], x1),
                max(current_bbox[3], y1),
            ]
        current_text.append(ch)
    flush_token()
    return tokens


def build_tesseract_tokens(
    image: np.ndarray,
    table_bbox_img: list[float] | None,
) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img_path = Path(tmp.name)
        Image.fromarray(image).save(img_path)

    try:
        proc = subprocess.run(
            [
                "tesseract",
                str(img_path),
                "stdout",
                "--psm",
                "6",
                "tsv",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Tesseract failed: {exc.stderr}") from exc
    finally:
        try:
            img_path.unlink(missing_ok=True)
        except Exception:
            pass

    lines = proc.stdout.splitlines()
    if not lines:
        return tokens
    header = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != len(header):
            continue
        text = parts[-1]
        if not text.strip():
            continue
        left, top, width, height = map(int, parts[6:10])
        x0 = float(left)
        y0 = float(top)
        x1 = float(left + width)
        y1 = float(top + height)
        if table_bbox_img is not None:
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            tbx0, tby0, tbx1, tby1 = table_bbox_img
            if not (tbx0 <= cx <= tbx1 and tby0 <= cy <= tby1):
                continue
        tokens.append(
            {
                "id": len(tokens) + 1,
                "text": text,
                "bbox": {"l": x0, "t": y0, "r": x1, "b": y1},
                "block_id": 0,
                "text_line_id": len(tokens),
                "indexInLine": 0,
                "confidence": float(parts[10]) / 100.0 if parts[10].isdigit() else 1.0,
                "word_in_presentation_ltr_order": text,
                "lang": "en",
            }
        )
    return tokens


def build_iocr_tokens(
    row: dict[str, Any],
    scale: float,
    page_height: float,
    origin_y: float = 0.0,
    pdf_bounds: tuple[float, float, float, float] | None = None,
) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for idx, cell in enumerate(row["html"]["cells"]):
        bbox = cell.get("bbox")
        text = rich_text(cell.get("tokens", []))
        if not bbox or not text.strip():
            continue
        if pdf_bounds is not None:
            bbox = clamp_bbox_pdf(bbox, pdf_bounds)
        x0, y0, x1, y1 = scale_bbox(bbox, scale, page_height, origin_y)
        tokens.append(
            {
                "id": len(tokens) + 1,
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
    return tokens


def text_from_v1_cell(cell: dict[str, Any]) -> str:
    text_boxes = cell.get("text_cell_bboxes") or []
    return "".join(str(box.get("token", "")) for box in text_boxes if box.get("token"))


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

    def append_rows(parts: list[str], row_indexes: list[int]) -> None:
        for row_idx in row_indexes:
            parts.append("<tr>")
            col_idx = 0
            while col_idx < num_cols:
                cell = by_start.get((row_idx, col_idx))
                if cell is None:
                    col_idx += 1
                    continue
                attrs: list[str] = []
                if int(cell["row_span"]) > 1:
                    attrs.append(f' rowspan="{int(cell["row_span"])}"')
                if int(cell["col_span"]) > 1:
                    attrs.append(f' colspan="{int(cell["col_span"])}"')
                parts.append(f"<td{''.join(attrs)}>{cell['text']}</td>")
                col_idx = int(cell["end_col"])
            parts.append("</tr>")

    parts = ["<table>"]
    if header_rows:
        parts.append("<thead>")
        append_rows(parts, sorted(header_rows))
        parts.append("</thead>")

    body_rows = [row_idx for row_idx in range(num_rows) if row_idx not in header_rows]
    if body_rows:
        parts.append("<tbody>")
        append_rows(parts, body_rows)
        parts.append("</tbody>")

    parts.append("</table>")
    return "".join(parts)


def html_from_v1_sequence_with_text(html_seq: list[str], tf_responses: list[dict[str, Any]]) -> str:
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
    match = re.search(rf'{name}="(\\d+)"', tag)
    return int(match.group(1)) if match else 1


def html_from_v1_sequence_with_cell_map(
    html_seq: list[str],
    tf_responses: list[dict[str, Any]],
) -> str:
    """Inject matched text into the predicted HTML sequence by row/column.

    The native V1 response omits empty cells from ``tf_responses`` while
    ``html_seq`` still contains their tags. A plain sequential zip therefore
    shifts text into the wrong cells. Matching by predicted grid coordinates
    keeps empty cells empty and preserves the model's native structure.
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


def normalize_tf_output(tf_output: dict[str, Any], html_mode: str) -> str:
    details = tf_output["predict_details"]
    prediction = details.get("prediction", {})
    tf_responses = tf_output.get("tf_responses", [])

    if html_mode == "html_seq":
        return ensure_html_table_document(
            html_from_v1_sequence_with_text(prediction.get("html_seq", []), tf_responses)
        )
    if html_mode == "html_seq_cellmap":
        return canonicalize_pubtabnet_sections(
            html_from_v1_sequence_with_cell_map(prediction.get("html_seq", []), tf_responses)
        )

    cells = []
    for cell in tf_responses:
        cells.append(
            {
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
    return ensure_html_table_document(
        html_from_normalized_cells(cells, details.get("num_rows"), details.get("num_cols"))
    )


def teds_scores(pred_html: str, gt_html: str) -> tuple[float, float]:
    return (
        float(TEDS(structure_only=False).evaluate(pred_html, gt_html)),
        float(TEDS(structure_only=True).evaluate(pred_html, gt_html)),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.table_jsonl.open()]
    if args.only_filenames:
        allow = {
            line.strip()
            for line in args.only_filenames.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        rows = [row for row in rows if row.get("filename") in allow]
    rows = [row for row in rows if row_matches_shape_filter(row, args)]
    subset = rows[args.offset : args.offset + args.limit]
    predictor = TFPredictor(load_predictor_config(), device=args.device, num_threads=args.num_threads)

    teds_values: list[float] = []
    teds_s_values: list[float] = []
    per_table: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for sample_idx, row in enumerate(subset, start=args.offset):
        pdf_path = args.pdf_dir / row["filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF for {row['filename']}: {pdf_path}")
        doc = pdfium.PdfDocument(str(pdf_path))
        page = doc[0]
        textpage = page.get_textpage()
        mediabox, cropbox = get_pdf_boxes(page)
        bounds_candidates = [mediabox, tuple(float(v) for v in row["bbox"][:4])]
        cells_bbox = table_bbox_from_cells(row)
        if cells_bbox is not None:
            bounds_candidates.append(tuple(float(v) for v in cells_bbox))
        render_bounds = union_pdf_bounds(bounds_candidates)
        image, page_size, mediabox, cropbox = render_pdf_page(
            page, args.render_scale, override_bounds=render_bounds
        )
        if image is None or page_size is None or mediabox is None or cropbox is None:
            raise RuntimeError(f"Empty render for {row['filename']}")
        page_width, page_height = page_size
        _, render_y0, _, _ = render_bounds
        table_bbox, bbox_mode = scale_bbox_with_fallback(
            row["bbox"],
            args.render_scale,
            page_width,
            page_height,
            origin_y=render_y0,
            pdf_bounds=render_bounds,
        )
        if not bbox_is_valid(table_bbox):
            fallback_bbox = table_bbox_from_cells(row)
            if fallback_bbox is not None:
                table_bbox, bbox_mode = scale_bbox_with_fallback(
                    fallback_bbox,
                    args.render_scale,
                    page_width,
                    page_height,
                    origin_y=render_y0,
                    pdf_bounds=render_bounds,
                )
                bbox_mode = f"cells_{bbox_mode}"
            if not bbox_is_valid(table_bbox):
                raise RuntimeError(f"Invalid table bbox for {row['filename']}: {table_bbox}")
        table_bbox_pdf = row.get("bbox") or cells_bbox
        if table_bbox_pdf is not None:
            table_bbox_pdf = clamp_bbox_pdf(table_bbox_pdf, render_bounds)
        if args.token_source == "pdftext":
            cropbox_x0, cropbox_y0, _, _ = cropbox
            mediabox_x0, mediabox_y0, _, _ = mediabox
            text_offset = (cropbox_x0 - mediabox_x0, cropbox_y0 - mediabox_y0)
            tokens = build_pdf_text_tokens(
                textpage,
                args.render_scale,
                page_width,
                page_height,
                render_y0,
                render_bounds,
                table_bbox,
                text_offset=text_offset,
            )
        elif args.token_source == "tesseract":
            tokens = build_tesseract_tokens(image, table_bbox)
        else:
            tokens = build_iocr_tokens(
                row,
                args.render_scale,
                page_height,
                origin_y=render_y0,
                pdf_bounds=render_bounds,
            )
        iocr_page = {
            "image": image,
            "png_image_fn": str(pdf_path),
            "tokens": tokens,
            "width": image.shape[1],
            "height": image.shape[0],
            "table_bboxes": [table_bbox],
        }
        raw_output = predictor.multi_table_predict(
            iocr_page,
            deepcopy([table_bbox]),
            do_matching=True,
            correct_overlapping_cells=False,
            sort_row_col_indexes=True,
        )[0]
        pred_html = normalize_tf_output(raw_output, args.html_mode)
        gt_html = html_from_fintabnet(row)
        teds, teds_s = teds_scores(pred_html, gt_html)
        teds_values.append(teds)
        teds_s_values.append(teds_s)
        per_table.append(
            {
                "sample_index": sample_idx,
                "filename": row["filename"],
                "table_id": row.get("table_id"),
                "teds": teds,
                "teds_s": teds_s,
                "num_tokens": len(iocr_page["tokens"]),
                "num_rows_pred": raw_output["predict_details"].get("num_rows"),
                "num_cols_pred": raw_output["predict_details"].get("num_cols"),
                "bbox_mode": bbox_mode,
                "token_source": args.token_source,
            }
        )

        if args.save_debug_html:
            debug_dir = args.output_dir / "debug_html"
            write_json(
                debug_dir / f"{sample_idx:06d}.json",
                {
                    "sample_index": sample_idx,
                    "filename": row["filename"],
                    "table_id": row.get("table_id"),
                    "teds": teds,
                    "teds_s": teds_s,
                    "table_bbox": table_bbox,
                    "page_size": page_size,
                    "mediabox": mediabox,
                    "cropbox": cropbox,
                    "image_shape": list(image.shape),
                    "pred_html": pred_html,
                    "gt_html": gt_html,
                    "predict_details": raw_output.get("predict_details", {}),
                    "tf_responses": raw_output.get("tf_responses", []),
                },
            )

    summary = {
        "dataset": "FinTabNet",
        "source": "kaggle:jiongjiong/fintabnet",
        "split": "val",
        "subset_size": len(subset),
        "evaluated": len(per_table),
        "render_scale": args.render_scale,
        "html_mode": args.html_mode,
        "mean_teds": statistics.mean(teds_values) if teds_values else None,
        "mean_teds_s": statistics.mean(teds_s_values) if teds_s_values else None,
        "notes": [
            "Smoke adaptation using Kaggle FinTabNet v1.0.0 PDFs rendered locally.",
            "PDF and cell bboxes are scaled by render_scale.",
        ],
    }
    write_json(args.output_dir / "per_table.json", per_table)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
