"""Build TFLOP-style aux JSON for PubTabNet v2 validation split.

TFLOP's test pipeline expects a JSON dict keyed by filename with:
  {"html": "<html>...</html>", "type": "simple|complex|unknown"}.

PubTabNet_2.0.0.jsonl stores HTML as a structured dict; this script converts
val rows into full HTML strings and saves a dict for TFLOPTestDataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
PUBTABNET_JSONL = BASE_DIR / "data" / "TFLOP-dataset" / "meta_data" / "PubTabNet_2.0.0.jsonl"
IMAGES_DIR = BASE_DIR / "data" / "TFLOP-dataset" / "images" / "validation"
DEFAULT_OUT = BASE_DIR / "data" / "TFLOP-dataset" / "meta_data" / "pubtabnet_val_aux.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=PUBTABNET_JSONL)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def rich_text(tokens: list[str]) -> str:
    return "".join(tokens or [])


def html_from_pubtabnet_row(row: dict[str, Any]) -> str:
    cells = row["html"]["cells"]
    cell_iter = iter(cells)
    parts: list[str] = ["<html><body><table>"]
    for token in row["html"]["structure"]["tokens"]:
        if token == "</td>":
            cell = next(cell_iter)
            parts.append(rich_text(cell.get("tokens", [])))
            parts.append(token)
        else:
            parts.append(token)
    parts.append("</table></body></html>")
    return "".join(parts)


def main() -> None:
    args = parse_args()
    output: dict[str, dict[str, Any]] = {}
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != "val":
                continue
            filename = row.get("filename")
            if filename is None:
                continue
            if not (args.images_dir / filename).exists():
                continue
            output[filename] = {
                "html": html_from_pubtabnet_row(row),
                "type": row.get("type", "unknown"),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    print(f"saved={args.output} entries={len(output)}")


if __name__ == "__main__":
    main()
