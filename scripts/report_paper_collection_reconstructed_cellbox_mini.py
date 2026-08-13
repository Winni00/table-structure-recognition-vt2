"""Compare Paper Collection OCR-style vs reconstructed-cellbox mini run."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
OCR_RUN = ROOT / "results/tflop_paper_collection_ocr_style_707"
CELLBOX_RUN = ROOT / "results/tflop_paper_collection_reconstructed_cellbox_promising7"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def score_map(path: Path) -> dict[str, dict]:
    rows = load_json(path)
    # TFLOP evaluate_ted.py writes list rows:
    # [filename, pred, answer, ..., teds_s, teds]
    out = {}
    for row in rows:
        if isinstance(row, list):
            out[row[0]] = {"teds_s": float(row[-2]), "teds": float(row[-1])}
        elif isinstance(row, dict):
            out[row["filename"]] = {"teds_s": float(row["teds_s"]), "teds": float(row["teds"])}
    return out


def main() -> None:
    manifest = load_json(CELLBOX_RUN / "manifest.json")
    cell_scores = score_map(CELLBOX_RUN / "ted_score_output.json")
    ocr_scores = {
        row["filename"]: {"teds_s": row["teds_s"], "teds": row["teds"]}
        for row in load_json(OCR_RUN / "report_readable" / "per_table_manifest.json")
    }
    rows = []
    for item in manifest:
        fn = item["filename"]
        ocr = ocr_scores.get(fn, {"teds_s": None, "teds": None})
        cell = cell_scores.get(fn, {"teds_s": None, "teds": None})
        rows.append(
            {
                "filename": fn,
                "ocr_style_teds_s": ocr["teds_s"],
                "ocr_style_teds": ocr["teds"],
                "reconstructed_cellbox_teds_s": cell["teds_s"],
                "reconstructed_cellbox_teds": cell["teds"],
                "delta_teds_s": None if ocr["teds_s"] is None or cell["teds_s"] is None else cell["teds_s"] - ocr["teds_s"],
                "delta_teds": None if ocr["teds"] is None or cell["teds"] is None else cell["teds"] - ocr["teds"],
                "num_reconstructed_text_cells": item["num_reconstructed_text_cells"],
            }
        )
    valid = [r for r in rows if r["delta_teds_s"] is not None]
    summary = {
        "samples": len(valid),
        "ocr_style_mean_teds_s": sum(r["ocr_style_teds_s"] for r in valid) / len(valid),
        "ocr_style_mean_teds": sum(r["ocr_style_teds"] for r in valid) / len(valid),
        "reconstructed_cellbox_mean_teds_s": sum(r["reconstructed_cellbox_teds_s"] for r in valid) / len(valid),
        "reconstructed_cellbox_mean_teds": sum(r["reconstructed_cellbox_teds"] for r in valid) / len(valid),
        "mean_delta_teds_s": sum(r["delta_teds_s"] for r in valid) / len(valid),
        "mean_delta_teds": sum(r["delta_teds"] for r in valid) / len(valid),
        "improved_teds_s": sum(1 for r in valid if r["delta_teds_s"] > 0),
        "improved_teds": sum(1 for r in valid if r["delta_teds"] > 0),
    }
    (CELLBOX_RUN / "comparison.json").write_text(
        json.dumps({"summary": summary, "per_table": rows}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Reconstructed Cellbox Mini Sanity Test",
        "",
        "This compares the same 7 promising Paper Collection samples under two input styles:",
        "",
        "- OCR-style: PSENet+MASTER text-region boxes/text",
        "- Reconstructed-cellbox style: approximate GT cell boxes + GT cell text",
        "",
        "Important: reconstructed cell boxes are approximate diagnostic boxes, not true annotations.",
        "",
        "## Summary",
        "",
        f"- Samples: {summary['samples']}",
        f"- OCR-style mean TEDS-S/TEDS: {summary['ocr_style_mean_teds_s']:.4f} / {summary['ocr_style_mean_teds']:.4f}",
        f"- Reconstructed-cellbox mean TEDS-S/TEDS: {summary['reconstructed_cellbox_mean_teds_s']:.4f} / {summary['reconstructed_cellbox_mean_teds']:.4f}",
        f"- Mean delta TEDS-S/TEDS: {summary['mean_delta_teds_s']:+.4f} / {summary['mean_delta_teds']:+.4f}",
        f"- Improved TEDS-S/TEDS samples: {summary['improved_teds_s']} / {summary['improved_teds']}",
        "",
        "## Per Table",
        "",
        "| filename | OCR TEDS-S | OCR TEDS | Cellbox TEDS-S | Cellbox TEDS | delta TEDS-S | delta TEDS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['filename']}` | {row['ocr_style_teds_s']:.4f} | {row['ocr_style_teds']:.4f} | "
            f"{row['reconstructed_cellbox_teds_s']:.4f} | {row['reconstructed_cellbox_teds']:.4f} | "
            f"{row['delta_teds_s']:+.4f} | {row['delta_teds']:+.4f} |"
        )
    (CELLBOX_RUN / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
