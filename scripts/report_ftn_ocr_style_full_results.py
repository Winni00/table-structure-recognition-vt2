"""Report FTN OCR-style results against existing FTN baselines."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_OCR_RUN = ROOT / "results/tflop_fintabnet_ocr_style_full_10622"
DEFAULT_ANNOTATION_RUN = ROOT / "results/tflop_fintabnet_full_annotation_excl_problem_pages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-run-dir", type=Path, default=DEFAULT_OCR_RUN)
    parser.add_argument("--annotation-run-dir", type=Path, default=DEFAULT_ANNOTATION_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_ted_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        row[0]: {
            "filename": row[0],
            "pred_html": row[1],
            "gt_html": row[2],
            "edit_distance": float(row[3]),
            "teds_s": float(row[4]),
            "teds": float(row[5]),
        }
        for row in rows
    }


def summarize(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    if not values:
        raise ValueError("No TEDS rows found")
    return {
        "samples": len(values),
        "teds_s": sum(v["teds_s"] for v in values) / len(values),
        "teds": sum(v["teds"] for v in values) / len(values),
        "teds_s_1": sum(abs(v["teds_s"] - 1.0) < 1e-12 for v in values),
        "teds_1": sum(abs(v["teds"] - 1.0) < 1e-12 for v in values),
    }


def load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_aux_rec_stats(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        aux_rec = pickle.load(f)
    return {
        "samples": len(aux_rec),
        "total_regions": sum(len(v) for v in aux_rec.values()),
        "empty_samples": sum(1 for v in aux_rec.values() if not v),
        "empty_sample_names": [k for k, v in aux_rec.items() if not v][:50],
    }


def compare(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    rows = []
    for filename in common:
        lrow = left[filename]
        rrow = right[filename]
        rows.append(
            {
                "filename": filename,
                f"{left_name}_teds_s": lrow["teds_s"],
                f"{left_name}_teds": lrow["teds"],
                f"{right_name}_teds_s": rrow["teds_s"],
                f"{right_name}_teds": rrow["teds"],
                "delta_teds_s": rrow["teds_s"] - lrow["teds_s"],
                "delta_teds": rrow["teds"] - lrow["teds"],
            }
        )
    if not rows:
        return {"common_samples": 0}
    return {
        "common_samples": len(rows),
        "mean_delta_teds_s": sum(r["delta_teds_s"] for r in rows) / len(rows),
        "mean_delta_teds": sum(r["delta_teds"] for r in rows) / len(rows),
        "right_better_teds_s": sum(r["delta_teds_s"] > 1e-12 for r in rows),
        "right_worse_teds_s": sum(r["delta_teds_s"] < -1e-12 for r in rows),
        "right_same_teds_s": sum(abs(r["delta_teds_s"]) <= 1e-12 for r in rows),
        "right_better_teds": sum(r["delta_teds"] > 1e-12 for r in rows),
        "right_worse_teds": sum(r["delta_teds"] < -1e-12 for r in rows),
        "right_same_teds": sum(abs(r["delta_teds"]) <= 1e-12 for r in rows),
        "largest_right_improvements_by_teds": sorted(rows, key=lambda r: r["delta_teds"], reverse=True)[:25],
        "largest_right_drops_by_teds": sorted(rows, key=lambda r: r["delta_teds"])[:25],
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or (args.ocr_run_dir / "analysis_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr_ted_path = args.ocr_run_dir / "ted_score_output.json"
    annotation_official_path = args.annotation_run_dir / "ted_score_output.json"
    annotation_canon_path = args.annotation_run_dir / "ted_score_output_ftn_canonicalized.json"

    if not ocr_ted_path.exists():
        raise FileNotFoundError(f"OCR-style TEDS output is not ready yet: {ocr_ted_path}")

    ocr_rows = load_ted_rows(ocr_ted_path)
    annotation_official_rows = load_ted_rows(annotation_official_path)
    annotation_canon_rows = (
        load_ted_rows(annotation_canon_path) if annotation_canon_path.exists() else None
    )

    report: dict[str, Any] = {
        "ocr_run_dir": str(args.ocr_run_dir),
        "annotation_run_dir": str(args.annotation_run_dir),
        "runs": {
            "ftn_ocr_style_official_tflop_eval": summarize(ocr_rows),
            "ftn_annotation_input_official_tflop_eval": summarize(annotation_official_rows),
        },
        "ocr_validation": load_json_if_exists(args.ocr_run_dir / "post_ocr_validation.json"),
        "ocr_merge_summary": load_json_if_exists(args.ocr_run_dir / "aux_rec_merge_summary.json"),
        "inference_merge_summary": load_json_if_exists(args.ocr_run_dir / "inference_merge_summary.json"),
        "aux_rec_stats": load_aux_rec_stats(args.ocr_run_dir / "aux_rec.pkl"),
        "comparisons": {
            "annotation_official_to_ocr_official": compare(
                annotation_official_rows,
                ocr_rows,
                "annotation_official",
                "ocr_official",
            )
        },
    }
    if annotation_canon_rows is not None:
        report["runs"]["ftn_annotation_input_ftn_canonicalized_diagnostic"] = summarize(
            annotation_canon_rows
        )
        report["comparisons"]["annotation_canonicalized_to_ocr_official"] = compare(
            annotation_canon_rows,
            ocr_rows,
            "annotation_canonicalized",
            "ocr_official",
        )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# FTN OCR-style Full Run Report",
        "",
        "## Scores",
        "",
        "| Run | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in report["runs"].items():
        lines.append(
            f"| {name} | {stats['samples']} | {pct(stats['teds_s'])} | {pct(stats['teds'])} | {stats['teds_s_1']} | {stats['teds_1']} |"
        )
    lines.extend(["", "## OCR Validation", ""])
    validation = report.get("ocr_validation") or {}
    if validation:
        lines.extend(
            [
                f"- OCR samples: {validation.get('samples')}",
                f"- Total OCR regions: {validation.get('total_regions')}",
                f"- Empty OCR samples: {validation.get('empty_rec_samples')}",
                f"- Invalid bbox regions: {validation.get('invalid_bbox_regions')}",
                f"- Out-of-bounds regions: {validation.get('out_of_bounds_regions')}",
            ]
        )
    lines.extend(["", "## Official Annotation Input vs OCR-style Input", ""])
    comp = report["comparisons"]["annotation_official_to_ocr_official"]
    if comp.get("common_samples"):
        lines.extend(
            [
                f"- Common samples: {comp['common_samples']}",
                f"- Mean OCR minus annotation delta TEDS-S: {pct(comp['mean_delta_teds_s'])}",
                f"- Mean OCR minus annotation delta TEDS: {pct(comp['mean_delta_teds'])}",
                f"- OCR better / worse / same by TEDS: {comp['right_better_teds']} / {comp['right_worse_teds']} / {comp['right_same_teds']}",
                "",
                "### Largest OCR improvements by TEDS",
                "",
                "| Filename | Delta TEDS | Annotation TEDS | OCR TEDS |",
                "|---|---:|---:|---:|",
            ]
        )
        for item in comp["largest_right_improvements_by_teds"][:10]:
            lines.append(
                f"| {item['filename']} | {pct(item['delta_teds'])} | {pct(item['annotation_official_teds'])} | {pct(item['ocr_official_teds'])} |"
            )
        lines.extend(["", "### Largest OCR drops by TEDS", "", "| Filename | Delta TEDS | Annotation TEDS | OCR TEDS |", "|---|---:|---:|---:|"])
        for item in comp["largest_right_drops_by_teds"][:10]:
            lines.append(
                f"| {item['filename']} | {pct(item['delta_teds'])} | {pct(item['annotation_official_teds'])} | {pct(item['ocr_official_teds'])} |"
            )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report_json": str(out_dir / "report.json"), "readme": str(out_dir / "README.md")}, indent=2))


if __name__ == "__main__":
    main()
