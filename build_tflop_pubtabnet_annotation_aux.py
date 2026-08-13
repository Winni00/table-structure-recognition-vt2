"""Build TFLOP aux inputs from PubTabNet cell annotations.

This is an oracle-input evaluation adapter for PubTabNet validation data: instead
of using OCR/PSE detection results, it feeds the annotated PubTabNet cell boxes
and cell text to TFLOP's test pipeline. This is useful for separating TFLOP's
structure model behavior from OCR/detection quality on validation data.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_JSONL = BASE_DIR / "data" / "TFLOP-dataset" / "meta_data" / "PubTabNet_2.0.0.jsonl"
DEFAULT_IMAGES = BASE_DIR / "data" / "TFLOP-dataset" / "images" / "validation"
DEFAULT_ERRONEOUS = BASE_DIR / "data" / "TFLOP-dataset" / "meta_data" / "erroneous_pubtabnet_data.json"
DEFAULT_OUTPUT = BASE_DIR / "results" / "tflop_pubtabnet_val_annotations_aux"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--erroneous-json",
        type=Path,
        default=DEFAULT_ERRONEOUS,
        help="TFLOP erroneous/ambiguous PubTabNet filename list to exclude.",
    )
    parser.add_argument(
        "--include-erroneous",
        action="store_true",
        help="Keep filenames listed in erroneous_pubtabnet_data.json.",
    )
    return parser.parse_args()


def rich_text(tokens: list[str]) -> str:
    return "".join(tokens or [])


def html_from_pubtabnet_row(row: dict[str, Any]) -> str:
    cells = row["html"]["cells"]
    cell_iter = iter(cells)
    parts: list[str] = ["<html><body><table>"]
    for token in row["html"]["structure"]["tokens"]:
        if token == "</td>":
            cell = next(cell_iter, {})
            parts.append(rich_text(cell.get("tokens", [])))
            parts.append(token)
        else:
            parts.append(token)
    parts.append("</table></body></html>")
    return "".join(parts)


def rec_from_pubtabnet_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    rec_items: list[dict[str, Any]] = []
    for cell in row["html"]["cells"]:
        bbox = cell.get("bbox")
        text = rich_text(cell.get("tokens", []))
        if not bbox or len(bbox) != 4:
            continue
        if not text:
            continue
        rec_items.append({"bbox": [float(x) for x in bbox], "text": text, "score": 1.0})
    return rec_items


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_key = "validation" if args.split == "val" else args.split
    excluded_filenames: set[str] = set()
    if not args.include_erroneous and args.erroneous_json.exists():
        erroneous = json.loads(args.erroneous_json.read_text(encoding="utf-8"))
        excluded_filenames = set(erroneous.get(split_key, []))

    aux_json: dict[str, dict[str, Any]] = {}
    aux_rec: dict[str, list[dict[str, Any]]] = {}
    skipped_missing_image = 0
    skipped_empty_rec = 0
    skipped_erroneous = 0
    seen = 0

    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != args.split:
                continue
            if seen < args.offset:
                seen += 1
                continue
            if args.limit is not None and len(aux_json) >= args.limit:
                break
            seen += 1

            filename = row.get("filename")
            if not filename:
                continue
            if filename in excluded_filenames:
                skipped_erroneous += 1
                continue
            if not (args.images_dir / filename).exists():
                skipped_missing_image += 1
                continue
            rec_items = rec_from_pubtabnet_row(row)
            if not rec_items:
                skipped_empty_rec += 1
                continue

            aux_json[filename] = {
                "html": html_from_pubtabnet_row(row),
                "type": row.get("type", "unknown"),
            }
            aux_rec[filename] = rec_items

    aux_json_path = args.output_dir / "aux.json"
    aux_rec_path = args.output_dir / "aux_rec.pkl"
    summary_path = args.output_dir / "summary.json"
    aux_json_path.write_text(json.dumps(aux_json, ensure_ascii=False), encoding="utf-8")
    with aux_rec_path.open("wb") as f:
        pickle.dump(aux_rec, f)

    summary = {
        "samples": len(aux_json),
        "split": args.split,
        "jsonl": str(args.jsonl),
        "images_dir": str(args.images_dir),
        "aux_json": str(aux_json_path),
        "aux_rec": str(aux_rec_path),
        "skipped_missing_image": skipped_missing_image,
        "skipped_empty_rec": skipped_empty_rec,
        "skipped_erroneous": skipped_erroneous,
        "erroneous_json": str(args.erroneous_json),
        "include_erroneous": args.include_erroneous,
        "adapter": "PubTabNet annotated cell bbox/text oracle input",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
