#!/usr/bin/env python3
"""Rescore paper-collection TFLOP predictions after GT HTML canonicalization.

This does not run model inference. It keeps ``pred_string`` unchanged and only
rewrites ``answer_string`` into a PubTabNet/TFLOP-friendlier form.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import sys
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Comment
from Levenshtein import distance
from tqdm import tqdm


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
TFLOP_REPO = ROOT / "repo" / "TFLOP_clean"
sys.path.insert(0, str(TFLOP_REPO))

from evaluate_ted import strip_html_contents  # noqa: E402
from tflop.evaluator import TEDS  # noqa: E402


DEFAULT_RUN_DIR = ROOT / "results/tflop_paper_collection_ocr_style_707"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=50)
    return parser.parse_args()


def wrap_html(html: str) -> str:
    html = html.strip().lstrip("\ufeff")
    if html.startswith("<html><body><table>") and html.endswith("</table></body></html>"):
        return html
    if html.startswith("<table") and html.endswith("</table>"):
        return f"<html><body>{html}</body></html>"
    return f"<html><body><table>{html}</table></body></html>"


def canonicalize_gt_html(html: str) -> str:
    """Convert publisher-style table HTML to a simpler PubTabNet-like form."""

    soup = BeautifulSoup(wrap_html(html), "lxml")

    for node in soup.find_all(string=lambda text: isinstance(text, Comment)):
        node.extract()

    # Remove style/layout attributes that are not meaningful for TEDS structure.
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr not in {"rowspan", "colspan"}:
                del tag.attrs[attr]

    # PubTabNet/TFLOP predictions mainly use td, not th. The header section is
    # still represented by thead, so preserving th as an extra tag mostly
    # measures publisher style rather than TSR quality.
    for th in soup.find_all("th"):
        th.name = "td"

    # Convert line breaks to spaces so that multi-line publisher cells are
    # compared as one cell content string.
    for br in soup.find_all("br"):
        br.replace_with(" ")

    # Remove empty formatting wrappers but keep their text.
    for tag_name in ["span", "font"]:
        for tag in soup.find_all(tag_name):
            tag.unwrap()

    # Normalize text whitespace inside cells.
    for cell in soup.find_all(["td", "th"]):
        # Preserve simple inline tags like i/sup if present, but strip direct
        # whitespace-only artifacts around text.
        if not cell.find(True):
            cell.string = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()

    table = soup.find("table")
    if table is None:
        return wrap_html("")
    return f"<html><body>{str(table)}</body></html>"


def official_evaluate_tuple(item: tuple[str, str, str]) -> tuple[Any, ...]:
    file_name, pred_string, gold_string = item
    ted_s = TEDS(structure_only=True, n_jobs=1)
    ted = TEDS(structure_only=False, n_jobs=1, ignore_nodes=["b"])

    refined_pred = wrap_html(pred_string)
    refined_gold = wrap_html(gold_string)
    try:
        refined_pred = strip_html_contents(refined_pred)
        refined_gold = strip_html_contents(refined_gold)
    except Exception:
        # Fall back to wrapped HTML when the official string normalizer cannot
        # parse publisher remnants. This is recorded indirectly by the score.
        pass

    try:
        ted_s_score = ted_s.evaluate(refined_pred, refined_gold)
    except Exception:
        ted_s_score = 0.0
    try:
        ted_score = ted.evaluate(refined_pred, refined_gold)
    except Exception:
        ted_score = 0.0

    edit_distance = distance(pred_string, gold_string) / max(
        len(pred_string), len(gold_string), 1
    )
    return file_name, pred_string, gold_string, edit_distance, ted_s_score, ted_score


def summarize(rows: list[tuple[Any, ...]], out_path: Path) -> dict[str, Any]:
    summary = {
        "output": str(out_path),
        "num_samples": len(rows),
        "teds_s": sum(row[-2] for row in rows) / len(rows) if rows else 0.0,
        "teds": sum(row[-1] for row in rows) / len(rows) if rows else 0.0,
        "teds_s_1": sum(row[-2] == 1.0 for row in rows),
        "teds_1": sum(row[-1] == 1.0 for row in rows),
    }
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def evaluate(items: list[tuple[str, str, str]], num_processes: int, batch_size: int) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for start in tqdm(range(0, len(items), batch_size), desc="TEDS rescore"):
        batch = items[start : start + batch_size]
        with multiprocessing.Pool(processes=num_processes) as pool:
            rows.extend(pool.map(official_evaluate_tuple, batch))
    rows.sort(key=lambda row: row[0])
    return rows


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = args.output_dir or (run_dir / "gt_canonicalized_rescore")
    out_dir.mkdir(parents=True, exist_ok=True)

    inference = json.loads((run_dir / "full_model_inference.json").read_text(encoding="utf-8"))
    canonicalized_inference = {}
    changes = []
    for file_name, payload in sorted(inference.items()):
        raw_gt = payload["answer_string"]
        canon_gt = canonicalize_gt_html(raw_gt)
        canonicalized_inference[file_name] = {
            **payload,
            "answer_string": canon_gt,
        }
        changes.append(
            {
                "filename": file_name,
                "raw_gt_len": len(raw_gt),
                "canonical_gt_len": len(canon_gt),
                "removed_style_attrs": raw_gt.count("style="),
                "th_to_td": raw_gt.lower().count("<th"),
                "br_removed": raw_gt.lower().count("<br"),
            }
        )

    (out_dir / "full_model_inference.json").write_text(
        json.dumps(canonicalized_inference, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "gt_canonicalization_changes.json").write_text(
        json.dumps(changes, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    items = [
        (key, value["pred_string"], value["answer_string"])
        for key, value in sorted(canonicalized_inference.items())
    ]
    rows = evaluate(items, args.num_processes, args.batch_size)
    score_path = out_dir / "ted_score_output.json"
    score_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    summary = summarize(rows, score_path)

    original_summary_path = run_dir / "ted_score_output.summary.json"
    original_summary = (
        json.loads(original_summary_path.read_text())
        if original_summary_path.is_file()
        else None
    )
    markdown = [
        "# Paper Collection GT-Canonicalized Rescore",
        "",
        "No model inference was rerun. Predictions are unchanged; only GT HTML was converted from publisher-style HTML toward a PubTabNet/TFLOP-friendly style.",
        "",
        "Canonicalization applied to GT:",
        "",
        "- remove publisher style/layout attributes",
        "- convert `<th>` to `<td>`",
        "- replace `<br/>` with spaces",
        "- unwrap simple `<span>`/`font` wrappers",
        "- normalize direct cell whitespace",
        "",
        "## Scores",
        "",
        "| Run | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| GT-canonicalized rescore | {summary['num_samples']} | {summary['teds_s']:.4f} | {summary['teds']:.4f} | {summary['teds_s_1']} | {summary['teds_1']} |",
        "",
        "Interpretation: this isolates how much score is caused by GT HTML style, without changing OCR, TFLOP prediction, or model weights.",
    ]
    if original_summary is not None:
        markdown.insert(
            -3,
            f"| Original run | {original_summary['num_samples']} | {original_summary['teds_s']:.4f} | {original_summary['teds']:.4f} | {original_summary['teds_s_1']} | {original_summary['teds_1']} |",
        )
    else:
        markdown.insert(
            -3,
            "| Original run | n/a | n/a | n/a | n/a | n/a |",
        )
    (out_dir / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
