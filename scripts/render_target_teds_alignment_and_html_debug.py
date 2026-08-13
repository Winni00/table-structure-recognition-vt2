#!/usr/bin/env python3
"""Render examples around a target TEDS score with boxes and HTML debug."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
sys.path.insert(0, str(ROOT / "scripts"))

from render_near_one_alignment_and_html_debug import (  # noqa: E402
    FTN_IMAGES,
    FTN_RUN,
    OUT_DIR as _NEAR_OUT,
    PTN_IMAGES,
    PTN_RUN,
    draw_alignment,
    ftn_canonicalize,
    html_page,
    load_inference,
    load_rec,
    ptn_canonicalize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, default=0.8)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "target_0p8_alignment_html_debug",
    )
    return parser.parse_args()


def select_target(ted_path: Path, target: float, n: int) -> list[dict]:
    rows = json.loads(ted_path.read_text(encoding="utf-8"))
    candidates = [
        {
            "filename": r[0],
            "edit_distance": r[3],
            "teds_s": float(r[4]),
            "teds": float(r[5]),
        }
        for r in rows
    ]
    candidates.sort(key=lambda x: (abs(x["teds"] - target), -x["teds_s"], x["filename"]))
    return candidates[:n]


def render_dataset(name, run_dir, images_dir, ted_file, canonicalize, out_dir, target, count):
    selected = select_target(run_dir / ted_file, target, count)
    inference = load_inference(run_dir)
    rec = load_rec(run_dir)
    results = []
    for rank, item in enumerate(selected, 1):
        fn = item["filename"]
        raw_pred = inference[fn]["pred_string"]
        raw_gt = inference[fn]["answer_string"]
        can_pred = canonicalize(raw_pred)
        can_gt = canonicalize(raw_gt)
        stem = f"{rank:02d}_{Path(fn).stem}"
        png_path = out_dir / f"{stem}_boxes_origin.png"
        html_path = out_dir / f"{stem}_html_raw_vs_canonical.html"
        from PIL import Image

        w, h = Image.open(images_dir / fn).size
        draw_alignment(
            images_dir / fn,
            rec.get(fn, []),
            f"{name} TEDS~{target:.2f} example {rank}: {fn}",
            f"TEDS-S={item['teds_s']:.6f}, TEDS={item['teds']:.6f}. Blue=table/crop boundary, green=text-region boxes, red=origin/axes.",
            png_path,
        )
        html_path.write_text(
            html_page(
                f"{name} HTML debug: {fn}",
                {
                    "Raw prediction HTML": raw_pred,
                    "Canonicalized prediction HTML": can_pred,
                    "Raw GT HTML": raw_gt,
                    "Canonicalized GT HTML": can_gt,
                },
                {
                    "filename": fn,
                    "TEDS-S": f"{item['teds_s']:.6f}",
                    "TEDS": f"{item['teds']:.6f}",
                    "target": target,
                    "box origin": "top-left of image/crop",
                    "table box": f"[0, 0, {w}, {h}]",
                    "text-region boxes": len(rec.get(fn, [])),
                },
            ),
            encoding="utf-8",
        )
        results.append({**item, "png": str(png_path), "html": str(html_path), "text_region_boxes": len(rec.get(fn, []))})
    return results


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "ptn_val": render_dataset(
            "PTN-Val annotation-input",
            PTN_RUN,
            PTN_IMAGES,
            "ted_score_output.json",
            ptn_canonicalize,
            args.output_dir / "ptn_val",
            args.target,
            args.count,
        ),
        "ftn_canonicalized": render_dataset(
            "FTN annotation-input canonicalized",
            FTN_RUN,
            FTN_IMAGES,
            "ted_score_output_ftn_canonicalized.json",
            ftn_canonicalize,
            args.output_dir / "ftn_canonicalized",
            args.target,
            args.count,
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# Target TEDS {args.target:.2f} Alignment and HTML Debug", ""]
    for section, items in report.items():
        lines.append(f"## {section}")
        for item in items:
            lines.append(f"- {item['filename']}: TEDS-S={item['teds_s']:.6f}, TEDS={item['teds']:.6f}")
            lines.append(f"  - PNG: `{item['png']}`")
            lines.append(f"  - HTML: `{item['html']}`")
        lines.append("")
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "readme": str(args.output_dir / "README.md")}, indent=2))


if __name__ == "__main__":
    main()
