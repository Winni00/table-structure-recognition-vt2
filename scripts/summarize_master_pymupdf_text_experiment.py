#!/usr/bin/env python3
"""Summarize a MASTER-vs-PyMuPDF Paper Collection experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


ROOT = Path("/cluster/home/trinhwin/vt2/docling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-run", type=Path, required=True)
    parser.add_argument("--pymupdf-run", type=Path, required=True)
    parser.add_argument("--error-classes", type=Path, default=ROOT / "results/tflop_paper_collection_updated_crops_ocr_style_public/error_mode_classification/records.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_summary(run_dir: Path, canonicalized: bool) -> dict:
    if canonicalized:
        path = run_dir / "gt_canonicalized_rescore" / "ted_score_output.summary.json"
    else:
        path = run_dir / "ted_score_output.summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_scores(run_dir: Path, canonicalized: bool) -> dict[str, dict[str, float]]:
    if canonicalized:
        path = run_dir / "gt_canonicalized_rescore" / "ted_score_output.json"
    else:
        path = run_dir / "ted_score_output.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row[0]: {"teds_s": float(row[-2]), "teds": float(row[-1])} for row in rows}


def load_error_classes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["filename"]: row["class"] for row in data.get("records", [])}


def class_stats(master_scores: dict, pymupdf_scores: dict, classes: dict[str, str]) -> dict:
    grouped: dict[str, list[dict[str, float]]] = {}
    for filename in sorted(set(master_scores) & set(pymupdf_scores)):
        cls = classes.get(filename, "other")
        grouped.setdefault(cls, []).append(
            {
                "delta_teds_s": pymupdf_scores[filename]["teds_s"] - master_scores[filename]["teds_s"],
                "delta_teds": pymupdf_scores[filename]["teds"] - master_scores[filename]["teds"],
            }
        )
    out = {}
    for cls, rows in grouped.items():
        out[cls] = {
            "samples": len(rows),
            "mean_delta_teds_s": mean(row["delta_teds_s"] for row in rows),
            "mean_delta_teds": mean(row["delta_teds"] for row in rows),
            "improved_teds": sum(1 for row in rows if row["delta_teds"] > 0.01),
            "worse_teds": sum(1 for row in rows if row["delta_teds"] < -0.01),
        }
    return dict(sorted(out.items()))


def top_deltas(master_scores: dict, pymupdf_scores: dict, classes: dict[str, str], n: int = 10) -> tuple[list[dict], list[dict]]:
    rows = []
    for filename in sorted(set(master_scores) & set(pymupdf_scores)):
        m = master_scores[filename]
        p = pymupdf_scores[filename]
        rows.append(
            {
                "filename": filename,
                "class": classes.get(filename, "other"),
                "master_teds_s": m["teds_s"],
                "master_teds": m["teds"],
                "pymupdf_teds_s": p["teds_s"],
                "pymupdf_teds": p["teds"],
                "delta_teds_s": p["teds_s"] - m["teds_s"],
                "delta_teds": p["teds"] - m["teds"],
            }
        )
    return sorted(rows, key=lambda r: r["delta_teds"], reverse=True)[:n], sorted(rows, key=lambda r: r["delta_teds"])[:n]


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master_official = load_summary(args.master_run, canonicalized=False)
    master_canon = load_summary(args.master_run, canonicalized=True)
    pymupdf_official = load_summary(args.pymupdf_run, canonicalized=False)
    pymupdf_canon = load_summary(args.pymupdf_run, canonicalized=True)
    master_scores = load_scores(args.master_run, canonicalized=True)
    pymupdf_scores = load_scores(args.pymupdf_run, canonicalized=True)
    classes = load_error_classes(args.error_classes)
    grouped = class_stats(master_scores, pymupdf_scores, classes)
    best, worst = top_deltas(master_scores, pymupdf_scores, classes)

    summary = {
        "master_run": str(args.master_run),
        "pymupdf_run": str(args.pymupdf_run),
        "official": {
            "master": master_official,
            "pymupdf": pymupdf_official,
            "delta_teds_s": pymupdf_official["teds_s"] - master_official["teds_s"],
            "delta_teds": pymupdf_official["teds"] - master_official["teds"],
        },
        "gt_canonicalized": {
            "master": master_canon,
            "pymupdf": pymupdf_canon,
            "delta_teds_s": pymupdf_canon["teds_s"] - master_canon["teds_s"],
            "delta_teds": pymupdf_canon["teds"] - master_canon["teds"],
        },
        "by_error_class_gt_canonicalized": grouped,
        "top_pymupdf_improvements": best,
        "top_pymupdf_drops": worst,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# MASTER vs PyMuPDF Text Extraction Summary",
        "",
        "Both runs use the same rotated table images, PSENet boxes, public TFLOP checkpoint, and TFLOP evaluation. Only the text source differs.",
        "",
        "## Overall Scores",
        "",
        "| Evaluation | Text source | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| Official | MASTER | {master_official['num_samples']} | {fmt(master_official['teds_s'])} | {fmt(master_official['teds'])} | {master_official['teds_s_1']} | {master_official['teds_1']} |",
        f"| Official | PyMuPDF | {pymupdf_official['num_samples']} | {fmt(pymupdf_official['teds_s'])} | {fmt(pymupdf_official['teds'])} | {pymupdf_official['teds_s_1']} | {pymupdf_official['teds_1']} |",
        f"| GT-canonicalized | MASTER | {master_canon['num_samples']} | {fmt(master_canon['teds_s'])} | {fmt(master_canon['teds'])} | {master_canon['teds_s_1']} | {master_canon['teds_1']} |",
        f"| GT-canonicalized | PyMuPDF | {pymupdf_canon['num_samples']} | {fmt(pymupdf_canon['teds_s'])} | {fmt(pymupdf_canon['teds'])} | {pymupdf_canon['teds_s_1']} | {pymupdf_canon['teds_1']} |",
        "",
        "## Delta by Previous Error Class",
        "",
        "| Class | Samples | Mean delta TEDS-S | Mean delta TEDS | Improved TEDS | Worse TEDS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls, row in grouped.items():
        lines.append(
            f"| {cls} | {row['samples']} | {fmt(row['mean_delta_teds_s'])} | {fmt(row['mean_delta_teds'])} | {row['improved_teds']} | {row['worse_teds']} |"
        )
    lines.extend(["", "## Largest PyMuPDF Improvements", "", "| File | Class | Delta TEDS-S | Delta TEDS |", "|---|---|---:|---:|"])
    for row in best:
        lines.append(f"| `{row['filename']}` | {row['class']} | {fmt(row['delta_teds_s'])} | {fmt(row['delta_teds'])} |")
    lines.extend(["", "## Largest PyMuPDF Drops", "", "| File | Class | Delta TEDS-S | Delta TEDS |", "|---|---|---:|---:|"])
    for row in worst:
        lines.append(f"| `{row['filename']}` | {row['class']} | {fmt(row['delta_teds_s'])} | {fmt(row['delta_teds'])} |")
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
