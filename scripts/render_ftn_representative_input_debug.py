#!/usr/bin/env python3
"""Render representative FTN input/debug samples by TEDS buckets."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/cluster/home/trinhwin/vt2/docling")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_tflop_fintabnet_smoke_aux import html_from_fintabnet  # noqa: E402
from render_near_one_alignment_and_html_debug import draw_alignment, html_page  # noqa: E402
from rescore_tflop_fintabnet_canonicalized import flatten_table_sections  # noqa: E402


RUN_DIR = ROOT / "results" / "tflop_fintabnet_full_annotation_excl_problem_pages"
IMAGES_DIR = RUN_DIR / "images"
FTN_JSONL = ROOT / "data" / "fintabnet_kaggle" / "FinTabNet_1.0.0_cell_val_excluding_28.jsonl"
OUT_DIR = ROOT / "results" / "ftn_representative_input_debug"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-per-bucket", type=int, default=5)
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


def clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value or "")
    return " ".join(value.split())


def rows_excerpt(value: str, max_rows: int = 10) -> str:
    rows = re.findall(r"<tr>.*?</tr>", value, flags=re.S)
    if len(rows) <= max_rows:
        selected = rows
    else:
        selected = rows[: max_rows // 2] + rows[-max_rows // 2 :]
    rendered = []
    for row in selected:
        rendered.append(row.replace("><", ">\n<"))
    if len(rows) > max_rows:
        rendered.insert(max_rows // 2, "... [middle rows omitted]")
    return "\n\n".join(rendered)


def load_original_gt_by_image() -> dict[str, dict[str, Any]]:
    by_table_id = {}
    with FTN_JSONL.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_table_id[str(row["table_id"])] = row
    result = {}
    for png in IMAGES_DIR.glob("fintabnet_*.png"):
        table_id = png.stem.split("_")[-1]
        if table_id in by_table_id:
            result[png.name] = by_table_id[table_id]
    return result


def select_buckets(count: int) -> dict[str, list[dict[str, Any]]]:
    rows = json.loads((RUN_DIR / "ted_score_output_ftn_canonicalized.json").read_text(encoding="utf-8"))
    records = []
    for file_name, pred, gt, edit, teds_s, teds in rows:
        records.append(
            {
                "file_name": file_name,
                "pred": pred,
                "gt": gt,
                "edit_distance": edit,
                "teds_s": float(teds_s),
                "teds": float(teds),
                "pred_stats": stats(pred),
                "gt_stats": stats(gt),
            }
        )
    perfect = [r for r in records if r["teds"] == 1.0]
    near = [r for r in records if 0.9 <= r["teds"] < 1.0]
    mid = [r for r in records if 0.7 <= r["teds"] < 0.9]
    bad = [r for r in records if r["teds"] < 0.5]
    perfect.sort(key=lambda r: r["file_name"])
    near.sort(key=lambda r: (-r["teds"], -r["teds_s"], r["file_name"]))
    mid.sort(key=lambda r: (abs(r["teds"] - 0.8), -r["teds_s"], r["file_name"]))
    bad.sort(key=lambda r: (r["teds"], r["teds_s"], r["file_name"]))
    return {
        "perfect_teds_1": perfect[:count],
        "near_perfect_0p9_1": near[:count],
        "middle_0p7_0p9": mid[:count],
        "bad_lt_0p5": bad[:count],
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inference = json.loads((RUN_DIR / "full_model_inference.json").read_text(encoding="utf-8"))
    with (RUN_DIR / "aux_rec.pkl").open("rb") as f:
        rec = pickle.load(f)
    original_gt = load_original_gt_by_image()
    buckets = select_buckets(args.count_per_bucket)

    manifest: dict[str, Any] = {
        "output_dir": str(args.out_dir),
        "run_dir": str(RUN_DIR),
        "canonicalized_scores": str(RUN_DIR / "ted_score_output_ftn_canonicalized.json"),
        "source_gt": str(FTN_JSONL),
        "buckets": {},
    }

    for bucket_name, bucket_records in buckets.items():
        out_bucket = args.out_dir / bucket_name
        out_bucket.mkdir(parents=True, exist_ok=True)
        manifest["buckets"][bucket_name] = []
        for idx, rec_item in enumerate(bucket_records, 1):
            fn = rec_item["file_name"]
            raw_pred = inference[fn]["pred_string"]
            raw_gt = inference[fn]["answer_string"]
            can_pred = flatten_table_sections(raw_pred)
            can_gt = flatten_table_sections(raw_gt)
            original_row = original_gt.get(fn)
            original_gt_html = html_from_fintabnet(original_row) if original_row else raw_gt
            text_regions = rec.get(fn, [])
            stem = f"{idx:02d}_{Path(fn).stem}"
            png_path = out_bucket / f"{stem}_crop_textboxes.png"
            html_path = out_bucket / f"{stem}_html_debug.html"
            draw_alignment(
                IMAGES_DIR / fn,
                text_regions,
                f"FTN input debug {bucket_name} #{idx}: {fn}",
                (
                    f"FTN-canonicalized TEDS-S={rec_item['teds_s']:.6f}, "
                    f"TEDS={rec_item['teds']:.6f}. Blue=crop boundary, green=text-region boxes."
                ),
                png_path,
            )
            html_path.write_text(
                html_page(
                    f"FTN representative input debug: {fn}",
                    {
                        "Original FTN GT HTML from JSONL": original_gt_html,
                        "Raw prediction HTML": raw_pred,
                        "Raw GT used by run": raw_gt,
                        "FTN-canonicalized prediction": can_pred,
                        "FTN-canonicalized GT": can_gt,
                        "Canonicalized row excerpt: prediction vs GT": (
                            "Prediction rows:\n"
                            + rows_excerpt(can_pred)
                            + "\n\nGT rows:\n"
                            + rows_excerpt(can_gt)
                        ),
                    },
                    {
                        "bucket": bucket_name,
                        "filename": fn,
                        "TEDS-S": f"{rec_item['teds_s']:.6f}",
                        "TEDS": f"{rec_item['teds']:.6f}",
                        "text regions": len(text_regions),
                        "prediction stats": rec_item["pred_stats"],
                        "GT stats": rec_item["gt_stats"],
                        "original GT found": original_row is not None,
                    },
                ),
                encoding="utf-8",
            )
            manifest["buckets"][bucket_name].append(
                {
                    "file_name": fn,
                    "teds_s": rec_item["teds_s"],
                    "teds": rec_item["teds"],
                    "text_regions": len(text_regions),
                    "pred_stats": rec_item["pred_stats"],
                    "gt_stats": rec_item["gt_stats"],
                    "png": str(png_path),
                    "html": str(html_path),
                    "original_gt_found": original_row is not None,
                }
            )

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = ["# FTN Representative Input Debug", ""]
    for bucket_name, items in manifest["buckets"].items():
        readme.append(f"## {bucket_name}")
        for item in items:
            readme.append(
                f"- `{item['file_name']}`: TEDS-S={item['teds_s']:.6f}, "
                f"TEDS={item['teds']:.6f}, text_regions={item['text_regions']}"
            )
            readme.append(f"  - PNG: `{item['png']}`")
            readme.append(f"  - HTML: `{item['html']}`")
        readme.append("")
    (args.out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "readme": str(args.out_dir / "README.md")}, indent=2))


if __name__ == "__main__":
    main()
