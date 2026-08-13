"""Summarize TFLOP full-run TEDS outputs when Slurm jobs finish."""

from __future__ import annotations

import json
from pathlib import Path


RUNS = [
    ("PTN val annotations", Path("results/tflop_pubtabnet_val_annotations_full")),
    ("PTN test official", Path("results/tflop_pubtabnet_test_9064_official_full_rerun")),
    ("FTN annotation excl problem pages", Path("results/tflop_fintabnet_full_annotation_excl_problem_pages")),
    ("FTN pdfplumber excl problem pages", Path("results/tflop_fintabnet_full_pdfplumber_excl_problem_pages")),
]


def summarize_ted(path: Path) -> dict | None:
    ted_path = path / "ted_score_output.json"
    if not ted_path.exists():
        return None
    rows = json.loads(ted_path.read_text(encoding="utf-8"))
    if not rows:
        return {"samples": 0}
    teds_s = [float(row[-2]) for row in rows]
    teds = [float(row[-1]) for row in rows]
    return {
        "samples": len(rows),
        "mean_teds_s": sum(teds_s) / len(teds_s),
        "mean_teds": sum(teds) / len(teds),
        "teds_s_eq_1": sum(abs(x - 1.0) < 1e-12 for x in teds_s),
        "teds_eq_1": sum(abs(x - 1.0) < 1e-12 for x in teds),
    }


def main() -> None:
    report = {}
    for name, path in RUNS:
        summary = summarize_ted(path)
        report[name] = summary if summary is not None else {"status": "not finished"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
