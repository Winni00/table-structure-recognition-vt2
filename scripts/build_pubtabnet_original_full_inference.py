#!/usr/bin/env python3
"""Build full_model_inference.json using original PubTabNet 2.0.0 GT HTML.

Inputs:
  - pubtabnet_jsonl: PubTabNet_2.0.0.jsonl with original GT content
  - pred_json: ted_score_output.json from TFLOP run (contains pred_string)
Output:
  - out_dir/full_model_inference.json compatible with TFLOP evaluate_ted.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pubtabnet-jsonl", type=Path, required=True)
    parser.add_argument("--pred-json-val", type=Path, required=True)
    parser.add_argument("--pred-json-test", type=Path, required=True)
    parser.add_argument("--out-dir-val", type=Path, required=True)
    parser.add_argument("--out-dir-test", type=Path, required=True)
    return parser.parse_args()


def ensure_html_table_document(text: str) -> str:
    cleaned = text.strip()
    if "<html" in cleaned.lower():
        return cleaned
    if "<table" in cleaned.lower():
        return f"<html><body>{cleaned}</body></html>"
    return f"<html><body><table>{cleaned}</table></body></html>"


def rich_text(tokens: list[str]) -> str:
    return "".join(tokens or [])


def html_from_pubtabnet(row: dict[str, Any]) -> str:
    html = row.get("html")
    if isinstance(html, str):
        return ensure_html_table_document(html)
    if not isinstance(html, dict):
        return ""
    structure = html.get("structure", {})
    tokens = structure.get("tokens", [])
    cells = html.get("cells", [])
    cell_iter = iter(cells)
    parts: list[str] = []
    for token in tokens:
        if token == "</td>":
            cell = next(cell_iter, {})
            parts.append(rich_text(cell.get("tokens", [])))
            parts.append(token)
        else:
            parts.append(token)
    return ensure_html_table_document("".join(parts))


def main() -> None:
    args = parse_args()
    args.out_dir_val.mkdir(parents=True, exist_ok=True)
    args.out_dir_test.mkdir(parents=True, exist_ok=True)

    pred_rows_val = json.loads(args.pred_json_val.read_text())
    pred_rows_test = json.loads(args.pred_json_test.read_text())
    target_names = {r[0] for r in pred_rows_val} | {r[0] for r in pred_rows_test}

    gold_map: dict[str, str] = {}
    with args.pubtabnet_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            fname = row.get("filename")
            if fname in target_names:
                gold_map[fname] = html_from_pubtabnet(row)
                if len(gold_map) == len(target_names):
                    break

    missing = target_names - set(gold_map.keys())
    if missing:
        (args.out_dir_val / "missing_filenames.txt").write_text(
            "\n".join(sorted(missing)),
            encoding="utf-8",
        )
        (args.out_dir_test / "missing_filenames.txt").write_text(
            "\n".join(sorted(missing)),
            encoding="utf-8",
        )

    def write_out(pred_rows, out_dir: Path) -> int:
        out = {}
        for fname, _, pred_html, *_ in pred_rows:
            gold = gold_map.get(fname)
            if not gold:
                continue
            out[fname] = {"pred_string": pred_html, "answer_string": gold}
        (out_dir / "full_model_inference.json").write_text(
            json.dumps(out, ensure_ascii=False),
            encoding="utf-8",
        )
        return len(out)

    n_val = write_out(pred_rows_val, args.out_dir_val)
    n_test = write_out(pred_rows_test, args.out_dir_test)
    print("written val", n_val, "test", n_test, "missing", len(missing))


if __name__ == "__main__":
    main()
