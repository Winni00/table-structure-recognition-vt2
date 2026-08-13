"""Evaluate the 12 hard-table benchmark across saved model output directories.

This script compares saved prediction artifacts against the shared hard-table
ground truth and produces per-table plus aggregated metrics for:

- TableFormer V1 standalone outputs
- TableFormer V2 standalone outputs
- TFLOP standalone outputs

Primary metrics:
- TEDS
- TEDS-S

Additional comparison-ready structure metrics:
- cell_precision_iou50
- cell_recall_iou50
- cell_f1_iou50

Why not a true AP50?
- The current benchmark artifacts contain HTML/XML table structure ground truth,
  but they do not provide image-space ground-truth cell boxes or per-cell model
  confidence scores.
- Because of that, a benchmark-faithful detection AP50 is not directly
  computable from the currently stored files.
- The script therefore records `ap50` as unavailable and computes a logical
  cell IoU@0.5 precision/recall/F1 on canonical table-grid cells instead.

The script can optionally write the computed `evaluation_result` back into each
prediction JSON so that raw output, normalized output, ground truth references
and evaluation live together in one artifact.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxml import etree, html


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_DATASET_JSON = (
    BASE_DIR / "data" / "paper_collection_meta" / "selected_hard_tables.json"
)
DEFAULT_MODEL_DIRS = {
    "tableformer_v1": BASE_DIR / "results" / "tableformer_hard_tables_v1",
    "tableformer_v2": BASE_DIR / "results" / "tableformer_hard_tables_v2",
    "tflop": BASE_DIR / "results" / "tflop_hard_tables",
}
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "evaluation"
TFLOP_REPO = BASE_DIR / "repo" / "TFLOP"


import sys

if str(TFLOP_REPO) not in sys.path:
    sys.path.insert(0, str(TFLOP_REPO))

from tflop.evaluator import TEDS  # noqa: E402


@dataclass
class LogicalCell:
    """Represents one canonical table cell on a logical row/column grid."""

    start_row: int
    end_row: int
    start_col: int
    end_col: int
    row_span: int
    col_span: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved hard-table predictions for TableFormer V1, "
            "TableFormer V2 and TFLOP."
        )
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_DATASET_JSON,
        help="Path to the selected hard tables metadata JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for benchmark summaries and per-model score files.",
    )
    parser.add_argument(
        "--write-back",
        action="store_true",
        help="Write evaluation_result back into each prediction JSON.",
    )
    return parser.parse_args()


def strip_xml_declarations(text: str) -> str:
    """Remove XML declarations that lxml.html does not need."""
    return re.sub(r"<\?xml[^>]*\?>", "", text).strip()


def ensure_html_table_document(text: str) -> str:
    """Wrap a table fragment into a full HTML document if needed."""
    cleaned = strip_xml_declarations(text)
    lower = cleaned.lower()
    if "<html" in lower:
        return cleaned
    if "<table" in lower:
        return f"<html><body>{cleaned}</body></html>"
    return f"<html><body><table>{cleaned}</table></body></html>"


def normalize_whitespace(text: str) -> str:
    """Collapse noisy whitespace while keeping textual content comparable."""
    return re.sub(r"\s+", " ", text).strip()


def canonical_html(text: str) -> str:
    """Return a stable HTML document for TEDS evaluation."""
    wrapped = ensure_html_table_document(text)
    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
    root = html.fromstring(wrapped, parser=parser)
    table_nodes = root.xpath("//table")
    if not table_nodes:
        return wrapped
    table = table_nodes[0]
    return html.tostring(
        html.fromstring(
            f"<html><body>{html.tostring(table, encoding='unicode')}</body></html>"
        ),
        encoding="unicode",
    )


def load_ground_truth_rows(dataset_json: Path) -> dict[str, dict[str, Any]]:
    """Load benchmark metadata keyed by table_id."""
    rows = json.loads(dataset_json.read_text(encoding="utf-8"))
    return {row["table_id"]: row for row in rows}


def prediction_files_by_sample(model_dir: Path) -> dict[str, Path]:
    """Index one result directory by sample id."""
    return {path.stem.replace(".pred", ""): path for path in model_dir.glob("*.pred.json")}


def unwrap_normalized_output(sample_json: dict[str, Any]) -> dict[str, Any]:
    """Handle the minor schema difference between TFLOP and TableFormer outputs."""
    normalized = sample_json["normalized_output"]
    if isinstance(normalized, list):
        if not normalized:
            return {"cells": [], "structure": {}}
        return normalized[0]
    return normalized


def parse_markup_to_logical_cells(markup: str) -> list[LogicalCell]:
    """Parse HTML/XML table markup into a canonical logical cell list."""
    wrapped = ensure_html_table_document(markup)
    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
    root = html.fromstring(wrapped, parser=parser)
    table_nodes = root.xpath("//table")
    if not table_nodes:
        return []
    table = table_nodes[0]

    rows = table.xpath("./thead/tr | ./tbody/tr | ./tfoot/tr | ./tr")
    occupied: set[tuple[int, int]] = set()
    parsed_cells: list[LogicalCell] = []

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row.xpath("./th | ./td"):
            while (row_idx, col_idx) in occupied:
                col_idx += 1

            colspan = int(cell.attrib.get("colspan", "1"))
            rowspan = int(cell.attrib.get("rowspan", "1"))
            start_col = col_idx
            end_col = col_idx + colspan - 1
            end_row = row_idx + rowspan - 1

            for rr in range(row_idx, end_row + 1):
                for cc in range(start_col, end_col + 1):
                    occupied.add((rr, cc))

            parsed_cells.append(
                LogicalCell(
                    start_row=row_idx,
                    end_row=end_row,
                    start_col=start_col,
                    end_col=end_col,
                    row_span=rowspan,
                    col_span=colspan,
                    text=normalize_whitespace("".join(cell.itertext())),
                )
            )
            col_idx = end_col + 1

    return parsed_cells


def extract_predicted_cells(
    sample_json: dict[str, Any],
    normalized: dict[str, Any],
) -> list[LogicalCell]:
    """Recover logical cells either from saved cell fields or from normalized HTML."""
    cells = normalized.get("cells") or []
    if cells:
        return [
            LogicalCell(
                start_row=int(cell["start_row"]),
                end_row=int(cell["end_row"]),
                start_col=int(cell["start_col"]),
                end_col=int(cell["end_col"]),
                row_span=int(cell["row_span"]),
                col_span=int(cell["col_span"]),
                text=normalize_whitespace(str(cell.get("text", ""))),
            )
            for cell in cells
        ]

    structure = normalized.get("structure", {})
    html_text = structure.get("html")
    if html_text:
        return parse_markup_to_logical_cells(html_text)

    raw_output = sample_json.get("raw_output", {})
    fallback_html = raw_output.get("full_pred_html") or raw_output.get("pred_string")
    if fallback_html:
        return parse_markup_to_logical_cells(fallback_html)
    return []


def extract_ground_truth_cells(gt_html_path: Path, gt_xml_path: Path | None) -> list[LogicalCell]:
    """Prefer GT HTML, then fall back to GT XML for logical-cell derivation."""
    if gt_html_path.exists():
        return parse_markup_to_logical_cells(gt_html_path.read_text(encoding="utf-8"))
    if gt_xml_path and gt_xml_path.exists():
        return parse_markup_to_logical_cells(gt_xml_path.read_text(encoding="utf-8"))
    return []


def logical_iou(a: LogicalCell, b: LogicalCell) -> float:
    """Compute IoU on logical table-grid boxes."""
    inter_w = max(0, min(a.end_col, b.end_col) - max(a.start_col, b.start_col) + 1)
    inter_h = max(0, min(a.end_row, b.end_row) - max(a.start_row, b.start_row) + 1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = (a.end_col - a.start_col + 1) * (a.end_row - a.start_row + 1)
    area_b = (b.end_col - b.start_col + 1) * (b.end_row - b.start_row + 1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def greedy_match_iou50(
    pred_cells: list[LogicalCell],
    gt_cells: list[LogicalCell],
) -> dict[str, float]:
    """Compute one-to-one logical-cell matching metrics at IoU >= 0.5."""
    if not pred_cells and not gt_cells:
        return {
            "cell_precision_iou50": 1.0,
            "cell_recall_iou50": 1.0,
            "cell_f1_iou50": 1.0,
            "matched_cells_iou50": 0,
            "pred_cell_count": 0,
            "gt_cell_count": 0,
        }

    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred_cell in enumerate(pred_cells):
        for gt_idx, gt_cell in enumerate(gt_cells):
            iou = logical_iou(pred_cell, gt_cell)
            if iou >= 0.5:
                candidates.append((iou, pred_idx, gt_idx))

    candidates.sort(reverse=True)
    used_preds: set[int] = set()
    used_gts: set[int] = set()
    matches = 0

    for _, pred_idx, gt_idx in candidates:
        if pred_idx in used_preds or gt_idx in used_gts:
            continue
        used_preds.add(pred_idx)
        used_gts.add(gt_idx)
        matches += 1

    precision = matches / len(pred_cells) if pred_cells else 0.0
    recall = matches / len(gt_cells) if gt_cells else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "cell_precision_iou50": precision,
        "cell_recall_iou50": recall,
        "cell_f1_iou50": f1,
        "matched_cells_iou50": matches,
        "pred_cell_count": len(pred_cells),
        "gt_cell_count": len(gt_cells),
    }


def safe_mean(values: list[float | None]) -> float | None:
    """Mean for non-null metric values."""
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def safe_median(values: list[float | None]) -> float | None:
    """Median for non-null metric values."""
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(statistics.median(valid))


def build_model_evaluator() -> tuple[TEDS, TEDS]:
    """Construct TEDS evaluators matching PubTabNet paper metric settings."""
    return (
        TEDS(structure_only=False, n_jobs=1),
        TEDS(structure_only=True, n_jobs=1),
    )


def evaluate_prediction(
    sample_json: dict[str, Any],
    gt_row: dict[str, Any],
    teds: TEDS,
    teds_s: TEDS,
) -> dict[str, Any]:
    """Evaluate one saved prediction artifact against GT HTML/XML."""
    normalized = unwrap_normalized_output(sample_json)
    structure = normalized.get("structure", {})

    pred_html = canonical_html(
        structure.get("html")
        or sample_json.get("raw_output", {}).get("full_pred_html")
        or sample_json.get("raw_output", {}).get("pred_string", "")
    )

    gt_html_path = Path(gt_row["gt_html"])
    gt_xml_path = Path(gt_row["gt_xml"]) if gt_row.get("gt_xml") else None
    gt_html = canonical_html(gt_html_path.read_text(encoding="utf-8"))

    pred_cells = extract_predicted_cells(sample_json, normalized)
    gt_cells = extract_ground_truth_cells(gt_html_path, gt_xml_path)
    logical_scores = greedy_match_iou50(pred_cells, gt_cells)

    return {
        "teds": teds.evaluate(pred_html, gt_html),
        "teds_s": teds_s.evaluate(pred_html, gt_html),
        "ap50": None,
        "ap50_status": (
            "not_available_with_current_artifacts:"
            "missing_ground_truth_cell_bboxes_and_prediction_confidences"
        ),
        **logical_scores,
        "ground_truth_source": {
            "gt_html": str(gt_html_path),
            "gt_xml": str(gt_xml_path) if gt_xml_path else None,
        },
    }


def evaluate_model_dir(
    *,
    model_name: str,
    model_dir: Path,
    gt_rows: dict[str, dict[str, Any]],
    output_dir: Path,
    write_back: bool,
) -> dict[str, Any]:
    """Evaluate all prediction files for one model directory."""
    teds, teds_s = build_model_evaluator()
    indexed_predictions = prediction_files_by_sample(model_dir)
    per_table_results: list[dict[str, Any]] = []

    for table_id, gt_row in gt_rows.items():
        pred_path = indexed_predictions.get(table_id)
        if pred_path is None:
            per_table_results.append(
                {
                    "sample": table_id,
                    "status": "missing_prediction",
                }
            )
            continue

        sample_json = json.loads(pred_path.read_text(encoding="utf-8"))
        evaluation_result = evaluate_prediction(sample_json, gt_row, teds, teds_s)
        sample_json["evaluation_result"] = evaluation_result

        if write_back:
            pred_path.write_text(
                json.dumps(sample_json, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        per_table_results.append(
            {
                "sample": table_id,
                "prediction_file": str(pred_path),
                "teds": evaluation_result["teds"],
                "teds_s": evaluation_result["teds_s"],
                "ap50": evaluation_result["ap50"],
                "cell_precision_iou50": evaluation_result["cell_precision_iou50"],
                "cell_recall_iou50": evaluation_result["cell_recall_iou50"],
                "cell_f1_iou50": evaluation_result["cell_f1_iou50"],
                "matched_cells_iou50": evaluation_result["matched_cells_iou50"],
                "pred_cell_count": evaluation_result["pred_cell_count"],
                "gt_cell_count": evaluation_result["gt_cell_count"],
                "ap50_status": evaluation_result["ap50_status"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_model_path = output_dir / f"{model_name}.per_table.json"
    per_model_path.write_text(
        json.dumps(per_table_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    complete_rows = [row for row in per_table_results if row.get("status") != "missing_prediction"]
    summary = {
        "model": model_name,
        "model_dir": str(model_dir),
        "tables_total": len(gt_rows),
        "tables_evaluated": len(complete_rows),
        "tables_missing_prediction": len(gt_rows) - len(complete_rows),
        "metrics": {
            "teds_mean": safe_mean([row.get("teds") for row in complete_rows]),
            "teds_median": safe_median([row.get("teds") for row in complete_rows]),
            "teds_s_mean": safe_mean([row.get("teds_s") for row in complete_rows]),
            "teds_s_median": safe_median([row.get("teds_s") for row in complete_rows]),
            "ap50_mean": safe_mean([row.get("ap50") for row in complete_rows]),
            "ap50_median": safe_median([row.get("ap50") for row in complete_rows]),
            "cell_precision_iou50_mean": safe_mean(
                [row.get("cell_precision_iou50") for row in complete_rows]
            ),
            "cell_recall_iou50_mean": safe_mean(
                [row.get("cell_recall_iou50") for row in complete_rows]
            ),
            "cell_f1_iou50_mean": safe_mean(
                [row.get("cell_f1_iou50") for row in complete_rows]
            ),
        },
        "notes": {
            "ap50": (
                "A true benchmark-style AP50 is not available from the current "
                "artifacts because GT cell bboxes and prediction confidences are "
                "not present."
            )
        },
        "per_table_file": str(per_model_path),
    }
    return summary


def main() -> None:
    args = parse_args()
    gt_rows = load_ground_truth_rows(args.dataset_json)

    benchmark_summary = []
    for model_name, model_dir in DEFAULT_MODEL_DIRS.items():
        benchmark_summary.append(
            evaluate_model_dir(
                model_name=model_name,
                model_dir=model_dir,
                gt_rows=gt_rows,
                output_dir=args.output_dir,
                write_back=args.write_back,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(benchmark_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Benchmark evaluation finished.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
