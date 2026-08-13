#!/usr/bin/env python3
"""Compare TFLOP predictions against original PubTabNet_2.0.0 GT HTML.

This script loads TFLOP prediction outputs (ted_score_output.json) for
PubTabNet val/test runs, replaces the GT with original PubTabNet HTML
from PubTabNet_2.0.0.jsonl, and recomputes TEDS / TEDS-S.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
TFLOP_REPO = (
    BASE_DIR / "repo" / "TFLOP_clean"
    if (BASE_DIR / "repo" / "TFLOP_clean").exists()
    else BASE_DIR / "repo" / "TFLOP"
)
if str(TFLOP_REPO) not in sys.path:
    sys.path.insert(0, str(TFLOP_REPO))

from tflop.evaluator import TEDS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pubtabnet-jsonl", type=Path, required=True)
    parser.add_argument("--val-preds", type=Path, required=True)
    parser.add_argument("--test-preds", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
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


def strip_html_contents(html_string: str) -> str:
    """Strip leading/trailing whitespace inside table cells for legacy comparisons."""
    if "<thead>" in html_string:
        pre_thead, remain_str = html_string.split("<thead>", 1)
    else:
        pre_thead = "<html><body><table>"
        remain_str = html_string[len(pre_thead) :]

    if "</thead>" in remain_str:
        thead, remain_str = remain_str.split("</thead>", 1)
        if not remain_str.startswith("<tbody>"):
            remain_str = "<tbody>" + remain_str
    else:
        if "<tbody>" in remain_str:
            thead, tmp_remain = remain_str.split("<tbody>", 1)
            remain_str = "<tbody>" + tmp_remain
        elif "</tbody>" in remain_str:
            thead, remain_str = remain_str.split("</tbody>", 1)
            remain_str = "<tbody></tbody>" + remain_str
        else:
            thead = remain_str.split("</table></body></html>")[0]
            remain_str = "<tbody></tbody></table></body></html>"

    remain_str = remain_str.split("<tbody>", 1)[1]
    if "</tbody>" in remain_str:
        tbody, post_tbody = remain_str.split("</tbody>", 1)
    else:
        if remain_str == "</table></body></html>":
            tbody = ""
            post_tbody = remain_str
        else:
            tbody = remain_str.split("</table></body></html>")[0]
            post_tbody = "</table></body></html>"

    thead_stripped, tbody_stripped = [], []
    thead_rows = thead.split("</tr>")
    for row in thead_rows:
        if row == "":
            continue
        if "<tr>" in row:
            row_contents = row.split("<tr>")[1].split("</td>")
        else:
            row_contents = row.split("</td>")

        row_stripped = []
        for row_content in row_contents:
            if row_content == "":
                continue
            td_header, td_content = row_content.split(">", 1)
            if td_content.startswith("<b>") and td_content.endswith("</b>"):
                td_content = td_content[3:-4].strip()
                td_content = "<b>" + td_content + "</b>"
            else:
                td_content = td_content.strip()
            new_td_entry = td_header + ">" + td_content + "</td>"
            row_stripped.append(new_td_entry)
        thead_stripped.append("<tr>" + "".join(row_stripped) + "</tr>")

    tbody_rows = tbody.split("</tr>")
    for row in tbody_rows:
        if row == "":
            continue
        if "<tr>" in row:
            row_contents = row.split("<tr>")[1].split("</td>")
        else:
            row_contents = row.split("</td>")

        row_stripped = []
        for row_content in row_contents:
            if row_content == "":
                continue
            td_header, td_content = row_content.split(">", 1)
            if td_content.startswith("<b>") and td_content.endswith("</b>"):
                td_content = td_content[3:-4].strip()
                td_content = "<b>" + td_content + "</b>"
            else:
                td_content = td_content.strip()
            new_td_entry = td_header + ">" + td_content + "</td>"
            row_stripped.append(new_td_entry)
        tbody_stripped.append("<tr>" + "".join(row_stripped) + "</tr>")

    new_html = (
        pre_thead
        + "<thead>"
        + "".join(thead_stripped)
        + "</thead><tbody>"
        + "".join(tbody_stripped)
        + "</tbody>"
        + post_tbody
    )
    return new_html


def normalize_html(html: str) -> str:
    refined = html
    if refined.startswith("<table>") and refined.endswith("</table>"):
        refined = "<html><body>" + refined + "</body></html>"
    elif not refined.startswith("<html><body><table>") and not refined.endswith(
        "</table></body></html>"
    ):
        refined = "<html><body><table>" + refined + "</table></body></html>"
    return refined


def load_pred_file(path: Path) -> list[list[Any]]:
    return json.loads(path.read_text())


def compute_scores(pred_rows: list[list[Any]], gold_map: dict[str, str]) -> dict[str, Any]:
    ted_struct = TEDS(structure_only=True, n_jobs=1)
    ted_full = TEDS(structure_only=False, n_jobs=1)

    scored = []
    missing = 0
    for row in pred_rows:
        fname, _, pred_html, _, _, _ = row
        gold_html = gold_map.get(fname)
        if not gold_html:
            missing += 1
            continue
        pred_norm = normalize_html(pred_html)
        gold_norm = normalize_html(gold_html)
        try:
            teds_s = ted_struct.evaluate(pred_norm, gold_norm)
        except Exception:
            teds_s = 0.0
        try:
            teds = ted_full.evaluate(pred_norm, gold_norm)
        except Exception:
            teds = 0.0
        scored.append([fname, gold_html, pred_html, teds, teds_s])

    if scored:
        mean_teds = sum(r[3] for r in scored) / len(scored)
        mean_teds_s = sum(r[4] for r in scored) / len(scored)
    else:
        mean_teds = 0.0
        mean_teds_s = 0.0

    return {
        "count_scored": len(scored),
        "count_missing": missing,
        "mean_teds": mean_teds,
        "mean_teds_s": mean_teds_s,
        "rows": scored,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    val_rows = load_pred_file(args.val_preds)
    test_rows = load_pred_file(args.test_preds)

    target_names = {r[0] for r in val_rows} | {r[0] for r in test_rows}

    gold_map: dict[str, str] = {}
    with args.pubtabnet_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            fname = row.get("filename")
            if fname in target_names:
                gold_map[fname] = html_from_pubtabnet(row)
                if len(gold_map) == len(target_names):
                    break

    val_scores = compute_scores(val_rows, gold_map)
    test_scores = compute_scores(test_rows, gold_map)

    (args.out_dir / "val_original_gt_summary.json").write_text(
        json.dumps({k: v for k, v in val_scores.items() if k != "rows"}, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "test_original_gt_summary.json").write_text(
        json.dumps({k: v for k, v in test_scores.items() if k != "rows"}, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "val_original_gt_scores.json").write_text(
        json.dumps(val_scores["rows"], indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "test_original_gt_scores.json").write_text(
        json.dumps(test_scores["rows"], indent=2),
        encoding="utf-8",
    )

    print("val:", val_scores["count_scored"], "missing:", val_scores["count_missing"], "mean_teds:", val_scores["mean_teds"], "mean_teds_s:", val_scores["mean_teds_s"])
    print("test:", test_scores["count_scored"], "missing:", test_scores["count_missing"], "mean_teds:", test_scores["mean_teds"], "mean_teds_s:", test_scores["mean_teds_s"])


if __name__ == "__main__":
    main()
