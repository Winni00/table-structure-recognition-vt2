#!/usr/bin/env python3
"""Render FTN cases where predictions differ from GT mainly by colspan/empty cells."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path("/cluster/home/trinhwin/vt2/docling")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_tflop_fintabnet_smoke_aux import html_from_fintabnet  # noqa: E402
from render_near_one_alignment_and_html_debug import draw_alignment, html_page  # noqa: E402
from rescore_tflop_fintabnet_canonicalized import flatten_table_sections  # noqa: E402


RUN_DIR = ROOT / "results" / "tflop_fintabnet_full_annotation_excl_problem_pages"
IMAGES_DIR = RUN_DIR / "images"
FTN_JSONL = ROOT / "data" / "fintabnet_kaggle" / "FinTabNet_1.0.0_cell_val_excluding_28.jsonl"
OUT_DIR = ROOT / "results" / "ftn_colspan_emptycell_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def stats(value: str) -> dict[str, int]:
    return {
        "tr": value.count("<tr>"),
        "td": value.count("<td"),
        "rowspan": value.count("rowspan="),
        "colspan": value.count("colspan="),
        "empty_td": len(re.findall(r"<td[^>]*>\s*</td>", value)),
        "thead": value.count("<thead>"),
        "tbody": value.count("<tbody>"),
    }


def load_original_gt_by_image() -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in FTN_JSONL.open(encoding="utf-8")]
    by_table_id = {str(row["table_id"]): row for row in rows}
    result = {}
    for png in IMAGES_DIR.glob("fintabnet_*.png"):
        stem = png.stem
        table_id = stem.split("_")[-1]
        row = by_table_id.get(table_id)
        if row:
            result[png.name] = row
    return result


def select_cases(count: int) -> list[dict[str, Any]]:
    rows = json.loads((RUN_DIR / "ted_score_output_ftn_canonicalized.json").read_text(encoding="utf-8"))
    cases = []
    for file_name, pred, gt, edit, teds_s, teds in rows:
        ps = stats(pred)
        gs = stats(gt)
        if ps["colspan"] == gs["colspan"]:
            continue
        cases.append(
            {
                "file_name": file_name,
                "pred": pred,
                "gt": gt,
                "edit_distance": edit,
                "teds_s": float(teds_s),
                "teds": float(teds),
                "pred_stats": ps,
                "gt_stats": gs,
                "colspan_delta": abs(ps["colspan"] - gs["colspan"]),
                "td_delta": abs(ps["td"] - gs["td"]),
                "empty_delta": abs(ps["empty_td"] - gs["empty_td"]),
            }
        )
    # Prefer representative cases where row count is equal, since those isolate span-vs-empty-cell style.
    cases.sort(
        key=lambda r: (
            r["pred_stats"]["tr"] != r["gt_stats"]["tr"],
            abs(r["teds"] - 0.75),
            -r["colspan_delta"],
            r["file_name"],
        )
    )
    return cases[:count]


def excerpt_rows(value: str, max_rows: int = 8) -> str:
    rows = re.findall(r"<tr>.*?</tr>", value, flags=re.S)
    if len(rows) <= max_rows:
        return "\n".join(row.replace("><", ">\n<") for row in rows)
    keep = rows[: max_rows // 2] + rows[-max_rows // 2 :]
    return "\n...\n".join(row.replace("><", ">\n<") for row in keep)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inference = json.loads((RUN_DIR / "full_model_inference.json").read_text(encoding="utf-8"))
    with (RUN_DIR / "aux_rec.pkl").open("rb") as f:
        rec = pickle.load(f)
    original_gt = load_original_gt_by_image()
    cases = select_cases(args.count)

    manifest = {"output_dir": str(args.out_dir), "cases": []}
    for idx, case in enumerate(cases, 1):
        fn = case["file_name"]
        raw_pred = inference[fn]["pred_string"]
        raw_gt = inference[fn]["answer_string"]
        original_row = original_gt.get(fn)
        original_gt_html = html_from_fintabnet(original_row) if original_row else raw_gt
        can_pred = flatten_table_sections(raw_pred)
        can_gt = flatten_table_sections(raw_gt)
        stem = f"{idx:02d}_{Path(fn).stem}"
        png_path = args.out_dir / f"{stem}_boxes.png"
        html_path = args.out_dir / f"{stem}_html.html"

        draw_alignment(
            IMAGES_DIR / fn,
            rec.get(fn, []),
            f"FTN colspan vs empty-cell case {idx}: {fn}",
            (
                f"Canonicalized TEDS-S={case['teds_s']:.6f}, TEDS={case['teds']:.6f}. "
                "Blue=crop boundary, green=FTN annotation-derived text boxes."
            ),
            png_path,
        )
        html_path.write_text(
            html_page(
                f"FTN colspan/empty-cell GT convention check: {fn}",
                {
                    "Original FTN GT HTML from JSONL": original_gt_html,
                    "Raw prediction HTML": raw_pred,
                    "Raw GT used by run": raw_gt,
                    "FTN-canonicalized prediction": can_pred,
                    "FTN-canonicalized GT": can_gt,
                    "Comparable row excerpts after canonicalization": (
                        "Prediction rows:\n"
                        + excerpt_rows(can_pred)
                        + "\n\nGT rows:\n"
                        + excerpt_rows(can_gt)
                    ),
                },
                {
                    "filename": fn,
                    "TEDS-S": f"{case['teds_s']:.6f}",
                    "TEDS": f"{case['teds']:.6f}",
                    "prediction stats": case["pred_stats"],
                    "GT stats": case["gt_stats"],
                    "colspan delta": case["colspan_delta"],
                    "td delta": case["td_delta"],
                    "empty cell delta": case["empty_delta"],
                    "original GT source": str(FTN_JSONL),
                },
            ),
            encoding="utf-8",
        )
        manifest["cases"].append(
            {
                **{k: v for k, v in case.items() if k not in {"pred", "gt"}},
                "png": str(png_path),
                "html": str(html_path),
                "original_gt_found": original_row is not None,
            }
        )

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = ["# FTN Colspan vs Empty-Cell Cases", ""]
    for case in manifest["cases"]:
        readme.append(
            f"- `{case['file_name']}`: TEDS-S={case['teds_s']:.6f}, TEDS={case['teds']:.6f}, "
            f"colspan_delta={case['colspan_delta']}, td_delta={case['td_delta']}, empty_delta={case['empty_delta']}"
        )
        readme.append(f"  - PNG: `{case['png']}`")
        readme.append(f"  - HTML: `{case['html']}`")
    (args.out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "cases": len(manifest["cases"])}, indent=2))


if __name__ == "__main__":
    main()
