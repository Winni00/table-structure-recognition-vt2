#!/usr/bin/env python3
"""Rescore TFLOP FinTabNet outputs after FTN-style HTML canonicalization.

FinTabNet annotations in this repo encode table rows directly below <table>
without PubTabNet-style <thead>/<tbody> wrappers. TFLOP predictions, however,
are emitted with <thead>/<tbody>. For a FinTabNet-style comparison we strip
these section wrappers from both prediction and ground truth before running the
same TEDS implementation.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from pathlib import Path

from Levenshtein import distance
from lxml import etree, html
from tqdm import tqdm


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
TFLOP_REPO = BASE_DIR / "repo" / "TFLOP"
sys.path.insert(0, str(TFLOP_REPO))

from tflop.evaluator import TEDS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=BASE_DIR / "results" / "tflop_fintabnet_full_annotation_excl_problem_pages",
    )
    parser.add_argument("--output-name", default="ted_score_output_ftn_canonicalized.json")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    return parser.parse_args()


def ensure_doc(html_string: str) -> str:
    html_string = html_string.strip()
    if html_string.startswith("<html"):
        return html_string
    if html_string.startswith("<table") and html_string.endswith("</table>"):
        return f"<html><body>{html_string}</body></html>"
    return f"<html><body><table>{html_string}</table></body></html>"


def flatten_table_sections(html_string: str) -> str:
    """Return HTML where all tr nodes are direct children of table."""
    parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
    doc = html.fromstring(ensure_doc(html_string), parser=parser)
    tables = doc.xpath("body/table")
    if not tables:
        return ensure_doc(html_string)
    table = tables[0]
    rows = []
    for child in list(table):
        if child.tag in {"thead", "tbody", "tfoot"}:
            rows.extend(list(child))
        elif child.tag == "tr":
            rows.append(child)
    for child in list(table):
        table.remove(child)
    for row in rows:
        table.append(row)
    return etree.tostring(doc, encoding="unicode", method="html")


def evaluate_one(item: tuple[str, str, str]) -> tuple:
    file_name, pred_string, gold_string = item
    refined_pred = flatten_table_sections(pred_string)
    refined_gold = flatten_table_sections(gold_string)
    edit_distance = distance(refined_pred, refined_gold) / max(
        len(refined_pred), len(refined_gold)
    )

    ted_s = TEDS(structure_only=True, n_jobs=1)
    ted = TEDS(structure_only=False, n_jobs=1)
    try:
        teds_s_score = ted_s.evaluate(refined_pred, refined_gold)
    except Exception:
        teds_s_score = 0.0
    try:
        teds_score = ted.evaluate(refined_pred, refined_gold)
    except Exception:
        teds_score = 0.0

    return (
        file_name,
        refined_pred,
        refined_gold,
        edit_distance,
        teds_s_score,
        teds_score,
    )


def main() -> None:
    args = parse_args()
    model_inference = json.loads(
        (args.run_dir / "full_model_inference.json").read_text(encoding="utf-8")
    )
    items = [
        (key, value["pred_string"], value["answer_string"])
        for key, value in sorted(model_inference.items(), key=lambda pair: pair[0])
    ]
    if args.limit is not None:
        items = items[: args.limit]
    if args.shard_index is not None or args.num_shards is not None:
        if args.shard_index is None or args.num_shards is None:
            raise ValueError("--shard-index and --num-shards must be set together")
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("--shard-index must be in [0, --num-shards)")
        items = [
            item
            for index, item in enumerate(items)
            if index % args.num_shards == args.shard_index
        ]

    outputs = []
    for start in tqdm(range(0, len(items), args.batch_size), desc="FTN canonical TEDS"):
        batch = items[start : start + args.batch_size]
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            outputs.extend(pool.map(evaluate_one, batch))

    out_path = args.run_dir / args.output_name
    out_path.write_text(json.dumps(outputs, ensure_ascii=False), encoding="utf-8")
    summary = {
        "run_dir": str(args.run_dir),
        "output": str(out_path),
        "num_samples": len(outputs),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "canonicalization": "flatten thead/tbody/tfoot so all tr nodes are direct table children",
        "teds_s": sum(row[-2] for row in outputs) / len(outputs),
        "teds": sum(row[-1] for row in outputs) / len(outputs),
        "teds_s_1": sum(row[-2] == 1.0 for row in outputs),
        "teds_1": sum(row[-1] == 1.0 for row in outputs),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
