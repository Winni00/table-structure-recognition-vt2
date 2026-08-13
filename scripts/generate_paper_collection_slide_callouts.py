#!/usr/bin/env python3
"""Generate slide-sized callouts for the paper-collection OCR-style run."""

from __future__ import annotations

import argparse
import html
import json
import pickle
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RUN_DIR = ROOT / "results/tflop_paper_collection_ocr_style_707"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "report_readable/callouts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_table_document(value: str) -> str:
    cleaned = value.strip()
    lower = cleaned.lower()
    if "<html" in lower:
        return cleaned
    if "<table" in lower:
        return f"<html><body>{cleaned}</body></html>"
    return f"<html><body><table>{cleaned}</table></body></html>"


def norm_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def short(value: str, limit: int = 110) -> str:
    value = norm_text(value)
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "..."


def parse_cells(value: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(ensure_table_document(value), "lxml")
    cells: list[dict[str, Any]] = []
    for row_idx, tr in enumerate(soup.find_all("tr")):
        for col_idx, cell in enumerate(tr.find_all(["td", "th"], recursive=False)):
            cells.append(
                {
                    "row": row_idx,
                    "col": col_idx,
                    "tag": cell.name,
                    "text": norm_text(cell.get_text(" ", strip=True)),
                    "rowspan": int(cell.get("rowspan") or 1),
                    "colspan": int(cell.get("colspan") or 1),
                    "has_style": bool(cell.get("style")),
                    "bold_count": len(cell.find_all("b")),
                    "italic_count": len(cell.find_all("i")),
                    "sup_count": len(cell.find_all("sup")),
                    "sub_count": len(cell.find_all("sub")),
                }
            )
    return cells


def table_stats(value: str) -> dict[str, Any]:
    soup = BeautifulSoup(ensure_table_document(value), "lxml")
    rows = soup.find_all("tr")
    cells = soup.find_all(["td", "th"])
    return {
        "rows": len(rows),
        "cells": len(cells),
        "thead": len(soup.find_all("thead")),
        "tbody": len(soup.find_all("tbody")),
        "th": len(soup.find_all("th")),
        "td": len(soup.find_all("td")),
        "bold": len(soup.find_all("b")),
        "italic": len(soup.find_all("i")),
        "sup": len(soup.find_all("sup")),
        "sub": len(soup.find_all("sub")),
        "styled_cells": sum(1 for cell in cells if cell.get("style")),
        "rowspan_cells": sum(1 for cell in cells if cell.get("rowspan")),
        "colspan_cells": sum(1 for cell in cells if cell.get("colspan")),
    }


def first_text_diffs(pred_cells: list[dict[str, Any]], gt_cells: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for idx, (pred, gt) in enumerate(zip(pred_cells, gt_cells)):
        if pred["text"] == gt["text"]:
            continue
        diffs.append(
            {
                "category": "OCR/text",
                "cell_index": idx,
                "row": pred["row"],
                "col": pred["col"],
                "finding": f"Cell r{pred['row']}c{pred['col']} text differs.",
                "prediction": short(pred["text"]),
                "ground_truth": short(gt["text"]),
            }
        )
        if len(diffs) >= limit:
            break
    return diffs


def structure_callout(pred_stats: dict[str, Any], gt_stats: dict[str, Any]) -> dict[str, Any]:
    row_delta = pred_stats["rows"] - gt_stats["rows"]
    cell_delta = pred_stats["cells"] - gt_stats["cells"]
    if row_delta or cell_delta:
        finding = (
            f"Predicted table shape differs by {row_delta:+d} rows and {cell_delta:+d} cells "
            f"({pred_stats['rows']}x/{pred_stats['cells']} vs {gt_stats['rows']}x/{gt_stats['cells']})."
        )
    else:
        finding = "Rows and cell counts match, so the remaining structure penalty is localized to cell semantics."
    return {
        "category": "structure",
        "finding": finding,
        "prediction": {"rows": pred_stats["rows"], "cells": pred_stats["cells"]},
        "ground_truth": {"rows": gt_stats["rows"], "cells": gt_stats["cells"]},
    }


def html_style_callout(pred_stats: dict[str, Any], gt_stats: dict[str, Any]) -> dict[str, Any]:
    parts: list[str] = []
    if pred_stats["th"] != gt_stats["th"]:
        parts.append(f"header cells `th` {pred_stats['th']} vs {gt_stats['th']}")
    if pred_stats["tbody"] != gt_stats["tbody"]:
        parts.append(f"`tbody` count {pred_stats['tbody']} vs {gt_stats['tbody']}")
    if pred_stats["bold"] != gt_stats["bold"]:
        parts.append(f"bold tags {pred_stats['bold']} vs {gt_stats['bold']}")
    if pred_stats["italic"] != gt_stats["italic"]:
        parts.append(f"italic tags {pred_stats['italic']} vs {gt_stats['italic']}")
    if pred_stats["sub"] != gt_stats["sub"] or pred_stats["sup"] != gt_stats["sup"]:
        parts.append(
            f"super/subscript tags {pred_stats['sup']}/{pred_stats['sub']} vs {gt_stats['sup']}/{gt_stats['sub']}"
        )
    if not parts:
        parts.append("no major tag-count difference")
    return {
        "category": "HTML-style",
        "finding": "Presentation/semantic markup differs: " + "; ".join(parts) + ".",
        "prediction": {
            "th": pred_stats["th"],
            "td": pred_stats["td"],
            "tbody": pred_stats["tbody"],
            "bold": pred_stats["bold"],
            "italic": pred_stats["italic"],
            "sup": pred_stats["sup"],
            "sub": pred_stats["sub"],
        },
        "ground_truth": {
            "th": gt_stats["th"],
            "td": gt_stats["td"],
            "tbody": gt_stats["tbody"],
            "bold": gt_stats["bold"],
            "italic": gt_stats["italic"],
            "sup": gt_stats["sup"],
            "sub": gt_stats["sub"],
        },
    }


def low_conf_ocr(aux_items: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(aux_items, key=lambda row: float(row.get("score", 1.0)))[:limit]:
        bbox = item.get("bbox")
        if bbox is not None:
            bbox = [round(float(v), 2) for v in bbox]
        rows.append(
            {
                "text": short(str(item.get("text", "")), 70),
                "score": round(float(item.get("score", 0.0)), 4),
                "bbox": bbox,
            }
        )
    return rows


def build_callouts(
    *,
    filename: str,
    bucket: str,
    manifest_row: dict[str, Any],
    ted_row: list[Any],
    inference_row: dict[str, Any],
    aux_rec: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    pred_html = inference_row["pred_string"]
    gt_html = inference_row["answer_string"]
    pred_cells = parse_cells(pred_html)
    gt_cells = parse_cells(gt_html)
    pred_stats = table_stats(pred_html)
    gt_stats = table_stats(gt_html)

    text_diffs = first_text_diffs(pred_cells, gt_cells, limit=1)
    if text_diffs:
        text_callout = text_diffs[0]
    else:
        text_callout = {
            "category": "OCR/text",
            "finding": "No same-position plain-text cell mismatch was found in the parsed preview.",
            "prediction": "",
            "ground_truth": "",
        }
    callouts = [text_callout, structure_callout(pred_stats, gt_stats), html_style_callout(pred_stats, gt_stats)]

    return {
        "bucket": bucket,
        "filename": filename,
        "teds": float(manifest_row["teds"]),
        "teds_s": float(manifest_row["teds_s"]),
        "ted_score_output_scores": {"teds_s": float(ted_row[-2]), "teds": float(ted_row[-1])},
        "teds_s_minus_teds": float(manifest_row["teds_s"]) - float(manifest_row["teds"]),
        "visualization": manifest_row["visualization"],
        "image": str(Path("results/tflop_paper_collection_ocr_style_707/images") / filename),
        "ocr_region_count": len(aux_rec.get(filename, [])),
        "lowest_confidence_ocr_regions": low_conf_ocr(aux_rec.get(filename, [])),
        "pred_stats": pred_stats,
        "gt_stats": gt_stats,
        "callouts": callouts,
    }


def selected_manifest_rows(example_manifest: dict[str, list[dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    wanted = [
        ("best_teds", 0),
        ("structure_ok_text_gap", 0),
        ("worst_teds", 0),
        ("worst_teds", 2),
    ]
    selected = []
    seen = set()
    for bucket, idx in wanted:
        row = example_manifest[bucket][idx]
        if row["filename"] in seen:
            continue
        seen.add(row["filename"])
        selected.append((bucket, row))
    return selected


def write_markdown(path: Path, examples: list[dict[str, Any]]) -> None:
    lines = [
        "# Paper-Collection OCR-Style Slide Callouts",
        "",
        "Small bounded callout set for the TFLOP OCR-style run.",
        "",
        "| Bucket | Example | TEDS-S | TEDS | Visualization |",
        "|---|---|---:|---:|---|",
    ]
    for ex in examples:
        lines.append(
            f"| {ex['bucket']} | `{ex['filename']}` | {ex['teds_s']:.4f} | {ex['teds']:.4f} | `{ex['visualization']}` |"
        )
    lines.extend(["", "## Callouts", ""])

    for ex in examples:
        lines.extend(
            [
                f"### {ex['bucket']}: `{ex['filename']}`",
                "",
                f"- Scores: TEDS-S `{ex['teds_s']:.4f}`, TEDS `{ex['teds']:.4f}`, gap `{ex['teds_s_minus_teds']:.4f}`",
                f"- Existing visualization: `{ex['visualization']}`",
                f"- Source image: `{ex['image']}`",
                f"- OCR regions from `aux_rec.pkl`: `{ex['ocr_region_count']}`",
                "",
            ]
        )
        for idx, callout in enumerate(ex["callouts"], start=1):
            lines.append(f"{idx}. **{callout['category']}**: {callout['finding']}")
            if "cell_index" in callout:
                lines.append(f"   - Predicted: `{callout['prediction']}`")
                lines.append(f"   - Ground truth: `{callout['ground_truth']}`")
        if ex["lowest_confidence_ocr_regions"]:
            low = ex["lowest_confidence_ocr_regions"][0]
            lines.append(
                f"- Lowest-confidence OCR region: `{low['text']}` at score `{low['score']}` bbox `{low['bbox']}`"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    example_manifest = load_json(run_dir / "report_readable/example_manifest.json")
    ted_rows = load_json(run_dir / "ted_score_output.json")
    inference = load_json(run_dir / "full_model_inference.json")
    ted_by_name = {row[0]: row for row in ted_rows}
    with (run_dir / "aux_rec.pkl").open("rb") as f:
        aux_rec = pickle.load(f)

    examples = []
    for bucket, manifest_row in selected_manifest_rows(example_manifest):
        filename = manifest_row["filename"]
        examples.append(
            build_callouts(
                filename=filename,
                bucket=bucket,
                manifest_row=manifest_row,
                ted_row=ted_by_name[filename],
                inference_row=inference[filename],
                aux_rec=aux_rec,
            )
        )

    (out_dir / "slide_callouts.json").write_text(
        json.dumps({"run_dir": str(run_dir), "examples": examples}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(out_dir / "slide_callouts.md", examples)
    print(json.dumps({"output_dir": str(out_dir), "examples": len(examples)}, indent=2))


if __name__ == "__main__":
    main()
