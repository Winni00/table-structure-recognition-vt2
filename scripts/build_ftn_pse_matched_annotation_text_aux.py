#!/usr/bin/env python3
"""Build FTN aux_rec with PSE boxes and annotation-matched text.

This creates the FinTabNet analogue of the PubTabNet train/val setup:
detected text-region boxes are used as model input, but their text is assigned
from the annotation cell that overlaps the detection.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_ANNOTATION_DIR = ROOT / "results/tflop_fintabnet_full_annotation_excl_problem_pages"
DEFAULT_PSE_DIR = ROOT / "results/tflop_fintabnet_ocr_style_full_10622"
DEFAULT_OUT_DIR = ROOT / "results/tflop_fintabnet_pse_matched_annotation_text_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--pse-dir", type=Path, default=DEFAULT_PSE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--iop-threshold", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def as_box(value: Any) -> tuple[float, float, float, float]:
    vals = [float(x) for x in list(value)]
    if len(vals) == 4:
        x1, y1, x2, y2 = vals
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    if len(vals) == 8:
        xs = vals[0::2]
        ys = vals[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    raise ValueError(f"Unsupported bbox length {len(vals)}: {vals[:8]}")


def area(box: tuple[float, float, float, float]) -> float:
    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)


def intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def match_box(
    pred_box: tuple[float, float, float, float],
    annotation_items: list[dict[str, Any]],
    iou_threshold: float,
    iop_threshold: float,
) -> tuple[dict[str, Any] | None, str | None, float]:
    pred_area = area(pred_box)
    best_iou_item = None
    best_iou = 0.0
    best_iop_item = None
    best_iop = 0.0

    for item in annotation_items:
        ann_box = as_box(item["bbox"])
        inter = intersection(pred_box, ann_box)
        if inter <= 0:
            continue
        union = pred_area + area(ann_box) - inter
        iou = inter / union if union > 0 else 0.0
        iop = inter / pred_area if pred_area > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_iou_item = item
        if iop > best_iop:
            best_iop = iop
            best_iop_item = item

    if best_iou_item is not None and best_iou >= iou_threshold:
        return best_iou_item, "iou", best_iou
    if best_iop_item is not None and best_iop >= iop_threshold:
        return best_iop_item, "iop", best_iop
    return None, None, 0.0


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aux = json.loads((args.annotation_dir / "aux.json").read_text(encoding="utf-8"))
    annotation_rec = load_pickle(args.annotation_dir / "aux_rec.pkl")
    pse_rec = load_pickle(args.pse_dir / "aux_rec.pkl")

    output_aux: dict[str, Any] = {}
    output_rec: dict[str, list[dict[str, Any]]] = {}
    per_file = []
    total_pse = 0
    total_matched = 0
    total_unmatched = 0
    total_iou = 0
    total_iop = 0
    empty_after_matching = []

    for filename in sorted(set(aux) & set(annotation_rec) & set(pse_rec)):
        ann_items = annotation_rec.get(filename, [])
        pse_items = pse_rec.get(filename, [])
        matched_items = []
        unmatched = 0
        match_modes = {"iou": 0, "iop": 0}

        for pse_item in pse_items:
            pse_box = as_box(pse_item["bbox"])
            ann_item, mode, score = match_box(
                pse_box,
                ann_items,
                args.iou_threshold,
                args.iop_threshold,
            )
            if ann_item is None:
                unmatched += 1
                continue
            match_modes[mode or "iou"] += 1
            matched_items.append(
                {
                    "bbox": [float(x) for x in pse_box],
                    "text": str(ann_item.get("text", "")),
                    "score": 1.0,
                    "bbox_score": float(pse_item.get("bbox_score", pse_item.get("score", 1.0))),
                    "match_mode": mode,
                    "match_score": float(score),
                    "source": "pse_box_with_annotation_text",
                }
            )

        total_pse += len(pse_items)
        total_matched += len(matched_items)
        total_unmatched += unmatched
        total_iou += match_modes["iou"]
        total_iop += match_modes["iop"]
        per_file.append(
            {
                "filename": filename,
                "annotation_regions": len(ann_items),
                "pse_regions": len(pse_items),
                "matched_regions": len(matched_items),
                "unmatched_pse_regions": unmatched,
                "iou_matches": match_modes["iou"],
                "iop_matches": match_modes["iop"],
            }
        )

        if matched_items:
            output_aux[filename] = aux[filename]
            output_rec[filename] = matched_items
        else:
            empty_after_matching.append(filename)

    (args.output_dir / "aux.json").write_text(
        json.dumps(output_aux, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.output_dir / "aux_rec.pkl").open("wb") as f:
        pickle.dump(output_rec, f)

    images_link = args.output_dir / "images"
    if images_link.exists() or images_link.is_symlink():
        images_link.unlink()
    images_link.symlink_to((args.annotation_dir / "images").resolve())

    summary = {
        "purpose": "FTN PSE-style boxes with annotation-matched GT text, analogous to PTN train/val PSE matching",
        "annotation_dir": str(args.annotation_dir),
        "pse_dir": str(args.pse_dir),
        "output_dir": str(args.output_dir),
        "iou_threshold": args.iou_threshold,
        "iop_threshold": args.iop_threshold,
        "input_samples_common": len(set(aux) & set(annotation_rec) & set(pse_rec)),
        "samples": len(output_aux),
        "dropped_empty_after_matching": len(empty_after_matching),
        "empty_after_matching_first": empty_after_matching[:50],
        "total_pse_regions": total_pse,
        "matched_regions": total_matched,
        "unmatched_pse_regions": total_unmatched,
        "match_rate": total_matched / total_pse if total_pse else 0.0,
        "iou_matches": total_iou,
        "iop_matches": total_iop,
        "mean_matched_regions_per_sample": total_matched / len(output_aux) if output_aux else 0.0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "matching_manifest.json").write_text(
        json.dumps(per_file, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
