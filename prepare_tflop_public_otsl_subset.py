"""
Prepare public FinTabNet/SynthTabNet subsets for TFLOP evaluation.

This script builds a TFLOP-compatible evaluation subset from public OTSL datasets
hosted on Hugging Face. It is intentionally an adaptation layer, not a strict
reproduction of the official TFLOP PubTabNet path. The official TFLOP release
ships PubTabNet test artifacts directly. For FinTabNet and SynthTabNet, we
derive TFLOP-style inputs from public image/html/cell annotations:

1. Save a subset of table images to a local directory.
2. Convert HTML annotations to TFLOP's `aux_json` structure.
3. Convert public cell annotations to TFLOP-style text-region entries and store
   them in a pickle matching the shape of `end2end_results.pkl`.

The resulting files can be used with TFLOP's original `test.py`:
  - images directory
  - aux JSON mapping filename -> {"html": ..., "type": ...}
  - OCR/text-region pickle mapping filename -> [{"bbox", "bbox_score", "text", "score"}, ...]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from datasets import load_dataset


SUPPORTED_DATASETS = {
    "fintabnet": "ds4sd/FinTabNet_OTSL",
    "synthtabnet": "ds4sd/SynthTabNet_OTSL",
}


def _flatten_token_entries(obj: Any) -> Iterator[dict[str, Any]]:
    """Yield token/bbox dictionaries from arbitrarily nested `cells` structures."""
    if isinstance(obj, dict):
        if "bbox" in obj and "tokens" in obj:
            yield obj
        for value in obj.values():
            yield from _flatten_token_entries(value)
        return

    if isinstance(obj, list):
        for item in obj:
            yield from _flatten_token_entries(item)


def _tokens_to_text(tokens: Any) -> str:
    """Convert token lists from the public OTSL datasets into a plain text string."""
    if isinstance(tokens, str):
        return tokens
    if isinstance(tokens, list):
        return "".join(str(tok) for tok in tokens)
    return str(tokens)


def _normalize_bbox(bbox: Any) -> list[float] | None:
    """Convert public bbox annotations into TFLOP-compatible [x1, y1, x2, y2]."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    # Public OTSL conversions often store [x1, y1, x2, y2, extra].
    x1, y1, x2, y2 = bbox[:4]
    return [float(x1), float(y1), float(x2), float(y2)]


def _html_value(row: dict[str, Any]) -> str:
    """Return a full HTML string with cell content from public OTSL rows."""
    cells = row.get("cells")
    otsl = row.get("otsl")
    if isinstance(cells, list) and isinstance(otsl, list):
        body_html = _cells_otsl_to_html(cells, otsl)
        return f"<html><body><table>{body_html}</table></body></html>"

    html = row.get("html")
    if isinstance(html, list):
        html = "".join(str(tok) for tok in html)
    if not isinstance(html, str):
        raise ValueError("Row does not contain a string-like `html` field.")

    if "<html>" not in html:
        html = f"<html><body><table>{html}</table></body></html>"
    return html


def _cells_otsl_to_html(cells: Any, otsl: list[Any]) -> str:
    """Reconstruct HTML with content from row-wise cell annotations and OTSL tags."""
    if not isinstance(cells, list):
        raise ValueError("Row does not contain list-like `cells` data.")
    if len(cells) == 1 and isinstance(cells[0], list):
        cells = cells[0]

    otsl_rows: list[list[str]] = []
    current_row: list[str] = []
    for token in otsl:
        token_str = str(token).strip().lower()
        if token_str == "nl":
            if current_row:
                otsl_rows.append(current_row)
            current_row = []
        else:
            current_row.append(token_str)
    if current_row:
        otsl_rows.append(current_row)

    if cells and isinstance(cells[0], dict):
        flat_cells = cells
        reshaped_cells: list[list[dict[str, Any]]] = []
        cursor = 0
        for row_tags in otsl_rows:
            real_cell_count = sum(1 for tag in row_tags if tag != "lcel")
            next_cursor = cursor + real_cell_count
            reshaped_cells.append(flat_cells[cursor:next_cursor])
            cursor = next_cursor
        cells = reshaped_cells

    if len(otsl_rows) != len(cells):
        raise ValueError(
            f"OTSL/cell row count mismatch: {len(otsl_rows)} vs {len(cells)}."
        )

    html_rows: list[str] = []
    for row_cells, row_tags in zip(cells, otsl_rows):
        if not isinstance(row_cells, list):
            raise ValueError("Expected each `cells` row to be a list.")
        expected_real_cells = sum(1 for tag in row_tags if tag != "lcel")
        if len(row_cells) != expected_real_cells:
            raise ValueError(
                f"OTSL/cell real-column count mismatch: {expected_real_cells} vs {len(row_cells)}."
            )

        td_parts: list[str] = []
        col_idx = 0
        cell_idx = 0
        while col_idx < len(row_tags):
            tag = row_tags[col_idx]
            if tag == "lcel":
                col_idx += 1
                continue

            if cell_idx >= len(row_cells):
                raise ValueError(
                    f"Row cell underflow: expected content cell for tag `{tag}`."
                )
            cell = row_cells[cell_idx]
            cell_idx += 1

            text = ""
            if isinstance(cell, dict):
                text = _tokens_to_text(cell.get("tokens", []))

            colspan = 1
            look_ahead = col_idx + 1
            while look_ahead < len(row_tags) and row_tags[look_ahead] == "lcel":
                colspan += 1
                look_ahead += 1

            attrs = f' colspan="{colspan}"' if colspan > 1 else ""
            td_parts.append(f"<td{attrs}>{text}</td>")
            col_idx = look_ahead

        html_rows.append(f"<tr>{''.join(td_parts)}</tr>")

    return "".join(html_rows)


def _html_type(html: str) -> str:
    """A lightweight type label mirroring TFLOP's official aux_json structure."""
    return "complex" if ("<thead>" in html or "<tbody>" in html) else "simple"


def _safe_filename(filename: str, fallback_index: int) -> str:
    """Ensure every saved example has a stable `.png` filename."""
    if not filename:
        return f"sample_{fallback_index:06d}.png"
    return filename if filename.endswith(".png") else f"{filename}.png"


def build_subset(
    dataset_name: str,
    split: str,
    sample_count: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Download a public OTSL subset and convert it to TFLOP evaluation inputs."""
    hf_name = SUPPORTED_DATASETS[dataset_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    subset = load_dataset(hf_name, split=f"{split}[:{sample_count}]")

    aux_json: dict[str, dict[str, str]] = {}
    aux_rec: dict[str, list[dict[str, Any]]] = {}
    rows_written = 0

    for idx, row in enumerate(subset):
        filename = _safe_filename(str(row.get("filename", "")).strip(), idx)
        image_path = image_dir / filename
        row["image"].save(image_path)

        html = _html_value(row)
        aux_json[filename] = {"html": html, "type": _html_type(html)}

        text_regions: list[dict[str, Any]] = []
        for entry in _flatten_token_entries(row.get("cells", [])):
            bbox = _normalize_bbox(entry.get("bbox"))
            text = _tokens_to_text(entry.get("tokens", []))
            if bbox is None or not text.strip():
                continue
            text_regions.append(
                {
                    "bbox": np.array(bbox, dtype=np.float32),
                    "bbox_score": np.float32(1.0),
                    "text": text,
                    "score": 1.0,
                }
            )

        aux_rec[filename] = text_regions
        rows_written += 1

    aux_json_path = output_dir / "aux_eval.json"
    aux_pkl_path = output_dir / "end2end_results.pkl"
    summary_path = output_dir / "summary.json"

    aux_json_path.write_text(json.dumps(aux_json, ensure_ascii=False), encoding="utf-8")
    with aux_pkl_path.open("wb") as fh:
        pickle.dump(aux_rec, fh)

    summary = {
        "dataset_name": dataset_name,
        "hf_dataset": hf_name,
        "split": split,
        "sample_count": rows_written,
        "image_dir": str(image_dir),
        "aux_json_path": str(aux_json_path),
        "aux_rec_pkl_path": str(aux_pkl_path),
        "mode": "adaptation",
        "notes": [
            "This subset is not an official Upstage TFLOP benchmark release.",
            "HTML comes from the public OTSL dataset.",
            "TFLOP-style text-region inputs are derived from public cell annotations.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(SUPPORTED_DATASETS),
        help="Public OTSL dataset to convert for TFLOP.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to sample from. Default: test",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1000,
        help="Number of examples to prepare. Default: 1000",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where images, aux JSON, OCR pickle and summary are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_subset(
        dataset_name=args.dataset,
        split=args.split,
        sample_count=args.sample_count,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
