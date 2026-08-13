"""Evaluate an automatic table-orientation detector for Paper Collection crops.

The detector is intentionally conservative.  It only decides between keeping the
crop as-is (0 degrees) and rotating it 270 degrees, because the manually reviewed
Paper Collection rotation labels only contain those two classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_IMAGE_DIR = ROOT / "results/tflop_paper_collection_updated_crops_ocr_style_public/images"
DEFAULT_LABELS = ROOT / "results/paper_collection_rotation_manual_review_31/rotation_labels.csv"
DEFAULT_OUT = ROOT / "results/paper_collection_auto_orientation_detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--manual-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.44)
    return parser.parse_args()


def orientation_score(image_path: Path) -> float:
    """Return a score where larger values mean the crop likely needs 270 deg rotation."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape
    scale = min(1.0, 1400 / max(h, w))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        h, w = img.shape

    _, bw = cv2.threshold(
        cv2.GaussianBlur(img, (3, 3), 0),
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Rotated table crops tend to have strong vertical line/text energy in the
    # unrotated image.  Normal horizontal tables tend to have stronger
    # horizontal row energy.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 18), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 18)))
    horizontal_energy = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel).sum() / 255
    vertical_energy = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel).sum() / 255

    row_projection = bw.sum(axis=1) / 255
    col_projection = bw.sum(axis=0) / 255
    row_peak = np.percentile(row_projection, 95) / (row_projection.mean() + 1e-6)
    col_peak = np.percentile(col_projection, 95) / (col_projection.mean() + 1e-6)

    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _, rotated_bw = cv2.threshold(
        cv2.GaussianBlur(rotated, (3, 3), 0),
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    rotated_row_projection = rotated_bw.sum(axis=1) / 255
    rotated_row_peak = np.percentile(rotated_row_projection, 95) / (
        rotated_row_projection.mean() + 1e-6
    )

    return (
        math.log((vertical_energy + 1) / (horizontal_energy + 1))
        + 0.5 * math.log((col_peak + 1e-6) / (row_peak + 1e-6))
        + 0.5 * math.log((rotated_row_peak + 1e-6) / (row_peak + 1e-6))
    )


def load_labels(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["filename"]: int(row["correct_rotation"]) for row in csv.DictReader(handle)}


def confusion(rows: list[dict]) -> dict:
    tp = sum(r["pred_rotation"] == 270 and r["manual_rotation"] == 270 for r in rows)
    tn = sum(r["pred_rotation"] == 0 and r["manual_rotation"] == 0 for r in rows)
    fp = sum(r["pred_rotation"] == 270 and r["manual_rotation"] == 0 for r in rows)
    fn = sum(r["pred_rotation"] == 0 and r["manual_rotation"] == 270 for r in rows)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(rows) if rows else None,
        "precision_rotate270": tp / (tp + fp) if tp + fp else None,
        "recall_rotate270": tp / (tp + fn) if tp + fn else None,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.manual_labels)
    rows: list[dict] = []
    for image_path in sorted(args.image_dir.glob("*.png")):
        score = orientation_score(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        rows.append(
            {
                "filename": image_path.name,
                "score": score,
                "pred_rotation": 270 if score > args.threshold else 0,
                "manual_rotation": labels.get(image_path.name),
                "width": width,
                "height": height,
                "aspect": width / height if height else None,
            }
        )

    labeled_rows = [r for r in rows if r["manual_rotation"] is not None]
    metrics = confusion(labeled_rows)
    summary = {
        "image_dir": str(args.image_dir),
        "manual_labels": str(args.manual_labels),
        "threshold": args.threshold,
        "all_samples": len(rows),
        "predicted_rotate270_all": sum(r["pred_rotation"] == 270 for r in rows),
        "manual_labelled_samples": len(labeled_rows),
        "manual_rotate270": sum(r["manual_rotation"] == 270 for r in labeled_rows),
        "manual_keep0": sum(r["manual_rotation"] == 0 for r in labeled_rows),
        **metrics,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (args.out_dir / "orientation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (args.out_dir / "rotation_labels_auto.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "correct_rotation", "notes"])
        writer.writeheader()
        for row in rows:
            if row["pred_rotation"] != 0:
                writer.writerow(
                    {
                        "filename": row["filename"],
                        "correct_rotation": row["pred_rotation"],
                        "notes": f"auto_orientation_score={row['score']:.3f}",
                    }
                )

    false_rows = [
        r
        for r in labeled_rows
        if r["pred_rotation"] != r["manual_rotation"]
    ]
    with (args.out_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# Paper Collection Orientation Detector\n\n")
        handle.write("Goal: detect whether a table crop should be rotated before PSENet/MASTER.\n\n")
        handle.write("The detector only predicts `0` or `270` degrees, matching the manual review labels.\n\n")
        handle.write("## Results on Manual Review Set\n\n")
        handle.write(f"- Manual samples: {summary['manual_labelled_samples']}\n")
        handle.write(f"- Manual rotate 270: {summary['manual_rotate270']}\n")
        handle.write(f"- Manual keep 0: {summary['manual_keep0']}\n")
        handle.write(f"- Accuracy: {summary['accuracy']:.3f}\n")
        handle.write(f"- Rotation precision: {summary['precision_rotate270']:.3f}\n")
        handle.write(f"- Rotation recall: {summary['recall_rotate270']:.3f}\n\n")
        handle.write("## Prediction on All Updated Crops\n\n")
        handle.write(f"- Samples: {summary['all_samples']}\n")
        handle.write(f"- Predicted 270 rotation: {summary['predicted_rotate270_all']}\n")
        handle.write(f"- Predicted keep 0: {summary['all_samples'] - summary['predicted_rotate270_all']}\n\n")
        handle.write("## Files\n\n")
        handle.write("- `orientation_predictions.csv`: score and prediction for every crop\n")
        handle.write("- `rotation_labels_auto.csv`: directly usable rotation labels for rotated cases\n\n")
        handle.write("## False Cases\n\n")
        if false_rows:
            for row in false_rows:
                handle.write(
                    f"- {row['filename']}: pred={row['pred_rotation']} manual={row['manual_rotation']} "
                    f"score={row['score']:.3f}\n"
                )
        else:
            handle.write("No mismatches on the manual review set.\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
