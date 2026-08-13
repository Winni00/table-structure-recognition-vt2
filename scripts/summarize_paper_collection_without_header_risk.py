#!/usr/bin/env python3
"""Summarize Paper Collection scores with and without TargetDomain header-risk PDFs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


HEADER_RISK_PAPERS = {
    "10.1016%j.bjid.2012.09.004",
    "10.1016%j.fm.2018.07.004",
    "10.1016%j.foodcont.2017.10.028",
    "10.1016%j.ijfoodmicro.2008.03.029",
    "10.1016%j.ijfoodmicro.2020.108750",
    "10.1016%j.meegid.2014.11.003",
    "10.1016%j.scitotenv.2004.09.025",
    "10.1016%j.vetmic.2011.11.009",
    "10.1016%j.vetmic.2014.02.045",
    "10.1016%j.vetmic.2017.06.010",
    "10.1186%s12866-017-0938-1",
    "10.1186%s12941-017-0242-9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path, paper_by_file: dict[str, str]) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        filename = row[0]
        paper_id = paper_by_file.get(filename, "")
        out.append(
            {
                "filename": filename,
                "paper_id": paper_id,
                "header_risk": paper_id in HEADER_RISK_PAPERS,
                "teds_s": float(row[-2]),
                "teds": float(row[-1]),
            }
        )
    return out


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"samples": 0, "teds_s": None, "teds": None, "teds_s_1": 0, "teds_1": 0}
    return {
        "samples": len(rows),
        "teds_s": mean(row["teds_s"] for row in rows),
        "teds": mean(row["teds"] for row in rows),
        "teds_s_1": sum(1 for row in rows if row["teds_s"] == 1.0),
        "teds_1": sum(1 for row in rows if row["teds"] == 1.0),
    }


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    paper_by_file = {row["filename"]: row["paper_id"] for row in manifest}
    risk_counts = Counter(row["paper_id"] for row in manifest if row["paper_id"] in HEADER_RISK_PAPERS)

    score_files = {
        "official": args.run_dir / "ted_score_output.json",
        "gt_canonicalized": args.run_dir / "gt_canonicalized_rescore" / "ted_score_output.json",
    }
    summary = {
        "run_dir": str(args.run_dir),
        "header_risk_papers": sorted(HEADER_RISK_PAPERS),
        "header_risk_counts": dict(sorted(risk_counts.items())),
        "evaluations": {},
    }
    for eval_name, score_file in score_files.items():
        rows = load_rows(score_file, paper_by_file)
        risk_rows = [row for row in rows if row["header_risk"]]
        clean_rows = [row for row in rows if not row["header_risk"]]
        summary["evaluations"][eval_name] = {
            "all": summarize(rows),
            "header_risk_only": summarize(risk_rows),
            "without_header_risk": summarize(clean_rows),
            "excluded_filenames": [row["filename"] for row in risk_rows],
        }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Paper Collection Header-Risk Sensitivity",
        "",
        "TargetDomain marked 12 PDFs as possible header-in-crop risk. This report recomputes the existing TEDS scores without those samples; no new inference was run.",
        "",
        "## Header-risk samples",
        "",
        "| PDF | Samples |",
        "|---|---:|",
    ]
    for paper, count in sorted(risk_counts.items()):
        lines.append(f"| `{paper}` | {count} |")
    lines.extend(["| **Total** | **{}** |".format(sum(risk_counts.values())), ""])

    for eval_name, data in summary["evaluations"].items():
        lines.extend(
            [
                f"## {eval_name}",
                "",
                "| Group | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for group in ["all", "header_risk_only", "without_header_risk"]:
            row = data[group]
            lines.append(
                f"| {group} | {row['samples']} | {fmt(row['teds_s'])} | {fmt(row['teds'])} | {row['teds_s_1']} | {row['teds_1']} |"
            )
        lines.append("")

    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
