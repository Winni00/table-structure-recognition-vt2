"""Build a tiny annotation-style TFLOP input from reconstructed cell boxes.

Uses the diagnostic output from
`feasibility_reconstruct_paper_collection_cell_boxes.py` and keeps only
examples labelled `promising`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RUN_DIR = ROOT / "results/tflop_paper_collection_ocr_style_707"
DEFAULT_FEAS_DIR = DEFAULT_RUN_DIR / "cell_box_reconstruction_feasibility"
DEFAULT_OUT_DIR = ROOT / "results/tflop_paper_collection_reconstructed_cellbox_promising7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--feasibility-dir", type=Path, default=DEFAULT_FEAS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--quality", default="promising")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(exist_ok=True)

    full_aux = json.loads((args.run_dir / "aux.json").read_text(encoding="utf-8"))
    debug = json.loads((args.feasibility_dir / "cell_assignment_debug.json").read_text(encoding="utf-8"))
    selected = [row for row in debug if row["quality"] == args.quality]
    if not selected:
        raise SystemExit(f"No examples found with quality={args.quality!r}")

    aux = {}
    aux_rec = {}
    manifest = []
    for row in selected:
        filename = row["filename"]
        aux[filename] = full_aux[filename]
        src_img = args.run_dir / "images" / filename
        shutil.copy2(src_img, image_dir / filename)
        rec_items = []
        for cell in row["cell_debug"]:
            text = cell.get("gt_text", "")
            if not text:
                continue
            bbox = cell.get("bbox", [])
            if len(bbox) != 4:
                continue
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            rec_items.append(
                {
                    "bbox": [float(v) for v in bbox],
                    "bbox_score": 1.0,
                    "text": text,
                    "score": 1.0,
                    "source": "reconstructed_gt_cellbox",
                }
            )
        aux_rec[filename] = rec_items
        manifest.append(
            {
                "filename": filename,
                "source_image": str(src_img),
                "num_reconstructed_text_cells": len(rec_items),
                "rows": row["rows"],
                "cols": row["cols"],
                "spans": row["spans"],
                "cell_coverage": row["cell_coverage"],
                "median_cell_text_similarity": row["median_cell_text_similarity"],
                "ocr_style_teds_s": row["teds_s"],
                "ocr_style_teds": row["teds"],
            }
        )

    (args.output_dir / "aux.json").write_text(json.dumps(aux, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "aux_rec.pkl").open("wb") as f:
        pickle.dump(aux_rec, f)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "subset.txt").write_text("\n".join(row["filename"] for row in manifest) + "\n", encoding="utf-8")
    summary = {
        "samples": len(manifest),
        "total_reconstructed_text_cells": sum(row["num_reconstructed_text_cells"] for row in manifest),
        "output_dir": str(args.output_dir),
        "note": "Tiny sanity test only: reconstructed boxes are approximate, not true annotations.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
