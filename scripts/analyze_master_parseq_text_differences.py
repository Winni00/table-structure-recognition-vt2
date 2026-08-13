#!/usr/bin/env python3
"""Analyze MASTER vs PARSeq text recognition on the same TFLOP inputs."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/cluster/home/trinhwin/vt2/docling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-run", type=Path, required=True)
    parser.add_argument("--parseq-run", type=Path, required=True)
    parser.add_argument(
        "--error-classes",
        type=Path,
        default=ROOT / "results/tflop_paper_collection_updated_crops_ocr_style_public/error_mode_classification/records.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary(run_dir: Path, canonicalized: bool) -> dict[str, Any]:
    path = run_dir / ("gt_canonicalized_rescore/ted_score_output.summary.json" if canonicalized else "ted_score_output.summary.json")
    return read_json(path)


def load_scores(run_dir: Path, canonicalized: bool) -> dict[str, dict[str, float]]:
    path = run_dir / ("gt_canonicalized_rescore/ted_score_output.json" if canonicalized else "ted_score_output.json")
    rows = read_json(path)
    return {row[0]: {"teds_s": float(row[-2]), "teds": float(row[-1])} for row in rows}


def load_aux(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("rb") as f:
        return pickle.load(f)


def load_error_classes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = read_json(path)
    return {row["filename"]: row["class"] for row in data.get("records", [])}


def summary_value(summary: dict[str, Any], key: str) -> Any:
    aliases = {
        "num_samples": ["num_samples", "samples"],
        "teds_s_1": ["teds_s_1", "teds-s=1", "teds_s_eq_1"],
        "teds_1": ["teds_1", "teds=1", "teds_eq_1"],
    }
    for candidate in aliases.get(key, [key]):
        if candidate in summary:
            return summary[candidate]
    return 0


def squash_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def no_ws(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def alnum_lower(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", (text or "").lower())


def confusion_norm(text: str) -> str:
    table = str.maketrans({
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
    })
    return alnum_lower((text or "").translate(table))


def classify_text_delta(master: str, parseq: str) -> str:
    m = master or ""
    p = parseq or ""
    if m == p:
        return "same"
    if squash_ws(m) == squash_ws(p):
        return "whitespace_only"
    if no_ws(m) == no_ws(p):
        return "spacing_only"
    if m.lower() == p.lower():
        return "case_only"
    if alnum_lower(m) == alnum_lower(p):
        return "punctuation_or_symbol_only"
    if confusion_norm(m) == confusion_norm(p):
        return "digit_letter_confusion_like"
    if not p.strip():
        return "parseq_empty"
    if not m.strip():
        return "master_empty"
    return "content_changed"


def region_examples(master_rows: list[dict[str, Any]], parseq_rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    examples = []
    for i, (mrow, prow) in enumerate(zip(master_rows, parseq_rows)):
        master_text = str(mrow.get("text", ""))
        parseq_text = str(prow.get("text", prow.get("parseq_text", "")))
        if master_text == parseq_text:
            continue
        examples.append(
            {
                "idx": i,
                "class": classify_text_delta(master_text, parseq_text),
                "master": master_text,
                "parseq": parseq_text,
                "master_score": float(mrow.get("score", 0.0) or 0.0),
                "parseq_score": float(prow.get("parseq_score", prow.get("score", 0.0)) or 0.0),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def global_region_stats(master_aux: dict[str, list[dict[str, Any]]], parseq_aux: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = Counter()
    changed_by_table = {}
    total = 0
    changed = 0
    for filename in sorted(set(master_aux) & set(parseq_aux)):
        table_changed = 0
        for mrow, prow in zip(master_aux[filename], parseq_aux[filename]):
            master_text = str(mrow.get("text", ""))
            parseq_text = str(prow.get("text", prow.get("parseq_text", "")))
            cls = classify_text_delta(master_text, parseq_text)
            counts[cls] += 1
            total += 1
            if cls != "same":
                changed += 1
                table_changed += 1
        changed_by_table[filename] = table_changed
    return {
        "total_regions": total,
        "changed_regions": changed,
        "changed_region_ratio": changed / total if total else 0.0,
        "classes": dict(counts.most_common()),
        "changed_regions_by_table": changed_by_table,
    }


def per_table_rows(
    master_scores: dict[str, dict[str, float]],
    parseq_scores: dict[str, dict[str, float]],
    master_aux: dict[str, list[dict[str, Any]]],
    parseq_aux: dict[str, list[dict[str, Any]]],
    classes: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for filename in sorted(set(master_scores) & set(parseq_scores) & set(master_aux) & set(parseq_aux)):
        m = master_scores[filename]
        p = parseq_scores[filename]
        changed = sum(
            1
            for mrow, prow in zip(master_aux[filename], parseq_aux[filename])
            if str(mrow.get("text", "")) != str(prow.get("text", prow.get("parseq_text", "")))
        )
        region_count = min(len(master_aux[filename]), len(parseq_aux[filename]))
        rows.append(
            {
                "filename": filename,
                "error_class": classes.get(filename, "other"),
                "master_teds_s": m["teds_s"],
                "master_teds": m["teds"],
                "parseq_teds_s": p["teds_s"],
                "parseq_teds": p["teds"],
                "delta_teds_s": p["teds_s"] - m["teds_s"],
                "delta_teds": p["teds"] - m["teds"],
                "regions": region_count,
                "changed_regions": changed,
                "changed_region_ratio": changed / region_count if region_count else 0.0,
                "examples": region_examples(master_aux[filename], parseq_aux[filename]),
            }
        )
    return rows


def grouped_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["error_class"]].append(row)
    out = {}
    for cls, cls_rows in grouped.items():
        out[cls] = {
            "samples": len(cls_rows),
            "mean_delta_teds_s": mean(row["delta_teds_s"] for row in cls_rows),
            "mean_delta_teds": mean(row["delta_teds"] for row in cls_rows),
            "improved_teds_gt_0_01": sum(row["delta_teds"] > 0.01 for row in cls_rows),
            "worse_teds_lt_minus_0_01": sum(row["delta_teds"] < -0.01 for row in cls_rows),
            "mean_changed_region_ratio": mean(row["changed_region_ratio"] for row in cls_rows),
        }
    return dict(sorted(out.items()))


def fmt(x: float) -> str:
    return f"{x:.4f}"


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def md_score_row(eval_name: str, source: str, summary: dict[str, Any]) -> str:
    return (
        f"| {eval_name} | {source} | {summary_value(summary, 'num_samples')} | "
        f"{fmt(float(summary['teds_s']))} | {fmt(float(summary['teds']))} | "
        f"{summary_value(summary, 'teds_s_1')} | {summary_value(summary, 'teds_1')} |"
    )


def md_delta_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| File | Class | MASTER TEDS | PARSeq TEDS | Delta TEDS | Changed regions | Examples |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        examples = "; ".join(
            f"{ex['class']}: `{ex['master']}` -> `{ex['parseq']}`" for ex in row["examples"][:3]
        )
        lines.append(
            f"| `{row['filename']}` | {row['error_class']} | {fmt(row['master_teds'])} | "
            f"{fmt(row['parseq_teds'])} | {fmt(row['delta_teds'])} | "
            f"{row['changed_regions']}/{row['regions']} | {examples} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master_aux = load_aux(args.master_run / "aux_rec.pkl")
    parseq_aux = load_aux(args.parseq_run / "aux_rec.pkl")
    classes = load_error_classes(args.error_classes)

    master_official = load_summary(args.master_run, canonicalized=False)
    parseq_official = load_summary(args.parseq_run, canonicalized=False)
    master_canon = load_summary(args.master_run, canonicalized=True)
    parseq_canon = load_summary(args.parseq_run, canonicalized=True)

    master_scores = load_scores(args.master_run, canonicalized=True)
    parseq_scores = load_scores(args.parseq_run, canonicalized=True)
    rows = per_table_rows(master_scores, parseq_scores, master_aux, parseq_aux, classes)

    region_stats = global_region_stats(master_aux, parseq_aux)
    by_class = grouped_stats(rows)
    improvements = sorted(rows, key=lambda row: row["delta_teds"], reverse=True)[: args.top_n]
    drops = sorted(rows, key=lambda row: row["delta_teds"])[: args.top_n]
    high_structure = [row for row in rows if row["master_teds_s"] >= 0.9 or row["parseq_teds_s"] >= 0.9]
    high_structure_improvements = sorted(high_structure, key=lambda row: row["delta_teds"], reverse=True)[: args.top_n]
    high_structure_drops = sorted(high_structure, key=lambda row: row["delta_teds"])[: args.top_n]

    summary = {
        "master_run": str(args.master_run),
        "parseq_run": str(args.parseq_run),
        "official": {
            "master": master_official,
            "parseq": parseq_official,
            "delta_parseq_minus_master_teds_s": parseq_official["teds_s"] - master_official["teds_s"],
            "delta_parseq_minus_master_teds": parseq_official["teds"] - master_official["teds"],
        },
        "gt_canonicalized": {
            "master": master_canon,
            "parseq": parseq_canon,
            "delta_parseq_minus_master_teds_s": parseq_canon["teds_s"] - master_canon["teds_s"],
            "delta_parseq_minus_master_teds": parseq_canon["teds"] - master_canon["teds"],
        },
        "table_delta_counts_gt_canonicalized": {
            "parseq_improves_teds_gt_0_01": sum(row["delta_teds"] > 0.01 for row in rows),
            "parseq_worse_teds_lt_minus_0_01": sum(row["delta_teds"] < -0.01 for row in rows),
            "roughly_same_abs_delta_le_0_01": sum(abs(row["delta_teds"]) <= 0.01 for row in rows),
        },
        "region_text_difference_stats": region_stats,
        "by_error_class_gt_canonicalized": by_class,
        "top_parseq_improvements": improvements,
        "top_parseq_drops": drops,
        "high_structure_top_parseq_improvements": high_structure_improvements,
        "high_structure_top_parseq_drops": high_structure_drops,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "per_table_deltas.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# MASTER vs PARSeq Text Recognition Analysis",
        "",
        "Both runs use the same updated target-domain crops, best-rotation selection, PSENet text-region boxes, public TFLOP checkpoint, and evaluator. Only the recognized text changes.",
        "",
        "## Overall Scores",
        "",
        "| Evaluation | Text source | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |",
        "|---|---|---:|---:|---:|---:|---:|",
        md_score_row("Official", "MASTER", master_official),
        md_score_row("Official", "PARSeq", parseq_official),
        md_score_row("GT-canonicalized", "MASTER", master_canon),
        md_score_row("GT-canonicalized", "PARSeq", parseq_canon),
        "",
        "## What Changed?",
        "",
        f"- PARSeq changed `{region_stats['changed_regions']}` / `{region_stats['total_regions']}` text regions ({pct(region_stats['changed_region_ratio'])}) compared to MASTER.",
        f"- On GT-canonicalized TEDS, PARSeq improved `{summary['table_delta_counts_gt_canonicalized']['parseq_improves_teds_gt_0_01']}` tables by more than 0.01 TEDS.",
        f"- It worsened `{summary['table_delta_counts_gt_canonicalized']['parseq_worse_teds_lt_minus_0_01']}` tables by more than 0.01 TEDS.",
        f"- `{summary['table_delta_counts_gt_canonicalized']['roughly_same_abs_delta_le_0_01']}` tables stayed roughly unchanged.",
        "",
        "## Region-Level Difference Types",
        "",
        "| Difference type | Count |",
        "|---|---:|",
    ]
    for cls, count in region_stats["classes"].items():
        lines.append(f"| {cls} | {count} |")
    lines.extend(
        [
            "",
            "## Delta by Previous Error Class",
            "",
            "| Class | Samples | Mean delta TEDS-S | Mean delta TEDS | PARSeq better | PARSeq worse | Mean changed regions |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cls, row in by_class.items():
        lines.append(
            f"| {cls} | {row['samples']} | {fmt(row['mean_delta_teds_s'])} | {fmt(row['mean_delta_teds'])} | "
            f"{row['improved_teds_gt_0_01']} | {row['worse_teds_lt_minus_0_01']} | {pct(row['mean_changed_region_ratio'])} |"
        )
    lines.extend(["", *md_delta_table("Largest PARSeq Improvements", improvements), "", *md_delta_table("Largest PARSeq Drops", drops)])
    lines.extend(
        [
            "",
            *md_delta_table("High-Structure Cases Where PARSeq Helps", high_structure_improvements),
            "",
            *md_delta_table("High-Structure Cases Where PARSeq Hurts", high_structure_drops),
            "",
            "## Interpretation",
            "",
            "- PARSeq is not broken: it returns text for all regions and sometimes improves individual tables.",
            "- Globally, however, MASTER still fits this dataset better in downstream TFLOP/TEDS.",
            "- Because there is no ground-truth text per PSENet box, region-level differences are not automatically correct or wrong. The TEDS delta tells us which recognizer helped the final table output.",
            "- The next useful direction is a hybrid or targeted recognizer test on the text/OCR-error subset, not replacing MASTER globally yet.",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "tables": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
