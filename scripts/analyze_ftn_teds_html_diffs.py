#!/usr/bin/env python3
"""Analyze FTN TEDS failures by comparing raw and TFLOP-normalized HTML."""

from __future__ import annotations

import argparse
import difflib
import html
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "tflop_fintabnet_full_annotation_excl_problem_pages_tflop_official_eval"
    / "ted_score_output.json"
)
DEFAULT_OUT = ROOT / "results" / "ftn_teds_html_diff_analysis"
OFFICIAL_EVAL = ROOT / "scripts" / "evaluate_ted_upstage_official.py"


def load_official_normalizer():
    spec = importlib.util.spec_from_file_location("eval_official", OFFICIAL_EVAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {OFFICIAL_EVAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.strip_html_contents


def wrap_like_official(value: str) -> str:
    refined = value
    if value.startswith("<table>") and value.endswith("</table>"):
        refined = "<html><body>" + value + "</body></html>"
    elif not value.startswith("<html><body><table>") and not value.endswith("</table></body></html>"):
        refined = "<html><body><table>" + refined + "</table></body></html>"
    return refined


def pretty(value: str) -> str:
    return value.replace("><", ">\n<")


def short(value: str, limit: int = 7000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n... [truncated]"


def section_row_counts(value: str) -> dict[str, int]:
    thead = ""
    tbody = ""
    m = re.search(r"<thead>(.*?)</thead>", value, flags=re.S)
    if m:
        thead = m.group(1)
    m = re.search(r"<tbody>(.*?)</tbody>", value, flags=re.S)
    if m:
        tbody = m.group(1)
    return {"thead_tr": thead.count("<tr>"), "tbody_tr": tbody.count("<tr>")}


def stats(value: str) -> dict[str, int]:
    section_counts = section_row_counts(value)
    result = {
        "thead": value.count("<thead>"),
        "tbody": value.count("<tbody>"),
        "tr": value.count("<tr>"),
        "td": value.count("<td"),
        "rowspan": value.count("rowspan="),
        "colspan": value.count("colspan="),
        "empty_td": len(re.findall(r"<td[^>]*>\s*</td>", value)),
    }
    result.update(section_counts)
    return result


def bucket(score: float) -> str:
    if score == 1.0:
        return "1.00"
    if score >= 0.9:
        return "0.90-0.99"
    if score >= 0.7:
        return "0.70-0.89"
    if score >= 0.5:
        return "0.50-0.69"
    return "<0.50"


def categories(raw_pred: str, raw_gt: str, norm_pred: str, norm_gt: str, teds_s: float, teds: float) -> list[str]:
    raw_pred_stats = stats(raw_pred)
    raw_gt_stats = stats(raw_gt)
    pred_stats = stats(norm_pred)
    gt_stats = stats(norm_gt)
    cats: list[str] = []

    if raw_pred_stats["thead"] or raw_pred_stats["tbody"] or raw_gt_stats["thead"] or raw_gt_stats["tbody"]:
        if (raw_pred_stats["thead"], raw_pred_stats["tbody"]) != (raw_gt_stats["thead"], raw_gt_stats["tbody"]):
            cats.append("raw_section_convention_mismatch")

    if pred_stats["thead_tr"] != gt_stats["thead_tr"] or pred_stats["tbody_tr"] != gt_stats["tbody_tr"]:
        cats.append("normalized_thead_tbody_split_mismatch")
    if pred_stats["tr"] != gt_stats["tr"]:
        cats.append("row_count_mismatch")
    if pred_stats["td"] != gt_stats["td"]:
        cats.append("cell_count_mismatch")
    if pred_stats["rowspan"] != gt_stats["rowspan"]:
        cats.append("rowspan_count_mismatch")
    if pred_stats["colspan"] != gt_stats["colspan"]:
        cats.append("colspan_count_mismatch")
    if pred_stats["empty_td"] != gt_stats["empty_td"]:
        cats.append("empty_cell_count_mismatch")
    if teds_s == 1.0 and teds < 1.0:
        cats.append("content_only_difference")
    if teds_s < 1.0:
        cats.append("structural_difference")
    if not cats:
        cats.append("uncategorized")
    return cats


def unified_diff(a: str, b: str, fromfile: str, tofile: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            pretty(a).splitlines(),
            pretty(b).splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
            n=4,
        )
    )


def table_row(cells: list[str]) -> str:
    return "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--examples-per-bucket", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    strip_html_contents = load_official_normalizer()
    rows = json.loads(args.input.read_text(encoding="utf-8"))

    records: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)

    for file_name, raw_pred, raw_gt, edit_distance, teds_s, teds in rows:
        norm_pred = strip_html_contents(wrap_like_official(raw_pred))
        norm_gt = strip_html_contents(wrap_like_official(raw_gt))
        rec = {
            "file_name": file_name,
            "edit_distance": edit_distance,
            "teds_s": teds_s,
            "teds": teds,
            "bucket": bucket(teds),
            "categories": categories(raw_pred, raw_gt, norm_pred, norm_gt, teds_s, teds),
            "raw_pred_stats": stats(raw_pred),
            "raw_gt_stats": stats(raw_gt),
            "norm_pred_stats": stats(norm_pred),
            "norm_gt_stats": stats(norm_gt),
            "raw_pred": raw_pred,
            "raw_gt": raw_gt,
            "norm_pred": norm_pred,
            "norm_gt": norm_gt,
        }
        records.append(rec)
        bucket_counts[rec["bucket"]] += 1
        for cat in rec["categories"]:
            category_counts[cat] += 1
            category_by_bucket[rec["bucket"]][cat] += 1

    summary = {
        "input": str(args.input),
        "samples": len(records),
        "mean_teds_s": sum(r["teds_s"] for r in records) / len(records),
        "mean_teds": sum(r["teds"] for r in records) / len(records),
        "bucket_counts": dict(bucket_counts),
        "category_counts": dict(category_counts),
        "category_by_bucket": {k: dict(v) for k, v in category_by_bucket.items()},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    slim_records = [
        {
            key: rec[key]
            for key in [
                "file_name",
                "edit_distance",
                "teds_s",
                "teds",
                "bucket",
                "categories",
                "raw_pred_stats",
                "raw_gt_stats",
                "norm_pred_stats",
                "norm_gt_stats",
            ]
        }
        for rec in records
    ]
    (args.out_dir / "records_summary.json").write_text(
        json.dumps(slim_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    examples: dict[str, list[dict[str, Any]]] = {}
    bucket_order = ["1.00", "0.90-0.99", "0.70-0.89", "0.50-0.69", "<0.50"]
    for name in bucket_order:
        bucket_records = [r for r in records if r["bucket"] == name]
        if name == "1.00":
            chosen = bucket_records[: args.examples_per_bucket]
        else:
            chosen = sorted(bucket_records, key=lambda r: (r["teds"], r["teds_s"]))[: args.examples_per_bucket]
        examples[name] = chosen

    example_json = {}
    for name, bucket_examples in examples.items():
        example_json[name] = []
        for rec in bucket_examples:
            example_json[name].append(
                {
                    "file_name": rec["file_name"],
                    "teds_s": rec["teds_s"],
                    "teds": rec["teds"],
                    "categories": rec["categories"],
                    "raw_diff": unified_diff(rec["raw_pred"], rec["raw_gt"], "raw_pred", "raw_gt"),
                    "normalized_diff": unified_diff(
                        rec["norm_pred"], rec["norm_gt"], "normalized_pred", "normalized_gt"
                    ),
                    "raw_pred": rec["raw_pred"],
                    "raw_gt": rec["raw_gt"],
                    "norm_pred": rec["norm_pred"],
                    "norm_gt": rec["norm_gt"],
                }
            )
    (args.out_dir / "examples_by_bucket.json").write_text(
        json.dumps(example_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>FTN TEDS HTML Diff Analysis</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.35}table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #aaa;padding:5px 8px;vertical-align:top}pre{white-space:pre-wrap;font-size:12px;background:#f6f6f6;border:1px solid #ccc;padding:8px;max-height:520px;overflow:auto}.tag{background:#eee;padding:1px 4px;margin-right:3px}h2{border-top:2px solid #222;padding-top:16px;margin-top:30px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.small{color:#555;font-size:13px}</style></head><body>",
        "<h1>FTN strict TFLOP official evaluator: HTML diff analysis</h1>",
        f"<p>Samples: <b>{summary['samples']}</b>; mean TEDS-S: <b>{summary['mean_teds_s']:.6f}</b>; mean TEDS: <b>{summary['mean_teds']:.6f}</b></p>",
        "<h2>Score buckets</h2><table><tr><th>Bucket</th><th>Count</th></tr>",
    ]
    for name in bucket_order:
        html_parts.append(table_row([name, bucket_counts.get(name, 0)]))
    html_parts.append("</table><h2>Error categories</h2><table><tr><th>Category</th><th>Count</th></tr>")
    for cat, count in category_counts.most_common():
        html_parts.append(table_row([cat, count]))
    html_parts.append("</table>")

    for bucket_name in bucket_order:
        html_parts.append(f"<h2>Examples: {html.escape(bucket_name)}</h2>")
        if not examples[bucket_name]:
            html_parts.append("<p>No examples.</p>")
            continue
        for rec in examples[bucket_name]:
            cats = " ".join(f"<span class='tag'>{html.escape(c)}</span>" for c in rec["categories"])
            html_parts.append(
                f"<h3>{html.escape(rec['file_name'])} | TEDS-S={rec['teds_s']:.6f}, TEDS={rec['teds']:.6f}</h3>"
            )
            html_parts.append(f"<p>{cats}</p>")
            html_parts.append("<div class='small'>Raw stats pred/gt: "
                              f"{html.escape(str(rec['raw_pred_stats']))} / {html.escape(str(rec['raw_gt_stats']))}<br>"
                              "Normalized stats pred/gt: "
                              f"{html.escape(str(rec['norm_pred_stats']))} / {html.escape(str(rec['norm_gt_stats']))}</div>")
            html_parts.append("<div class='grid'><div><b>Raw diff</b><pre>")
            html_parts.append(html.escape(short(unified_diff(rec["raw_pred"], rec["raw_gt"], "raw_pred", "raw_gt"))))
            html_parts.append("</pre></div><div><b>Normalized diff</b><pre>")
            html_parts.append(
                html.escape(short(unified_diff(rec["norm_pred"], rec["norm_gt"], "normalized_pred", "normalized_gt")))
            )
            html_parts.append("</pre></div></div>")
    html_parts.append("</body></html>")
    (args.out_dir / "report.html").write_text("\n".join(html_parts), encoding="utf-8")

    print(json.dumps({"out_dir": str(args.out_dir), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
