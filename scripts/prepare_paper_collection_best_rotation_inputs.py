"""Build a full paper-collection TFLOP input set with selected rotations.

The rotation-candidate experiment tested 0/90/270 degree variants for likely
rotated tables. This script creates a full run directory where only the
candidates that improved are replaced by their best rotated image. All other
tables keep the original updated TargetDomain crop.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_BASE_RUN = ROOT / "results/tflop_paper_collection_updated_crops_ocr_style_public"
DEFAULT_ROT_RUN = ROOT / "results/tflop_paper_collection_updated_crops_rotation_candidates_public"
DEFAULT_OUT_DIR = ROOT / "results/tflop_paper_collection_updated_crops_bestrot_master_public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument("--rotation-run-dir", type=Path, default=DEFAULT_ROT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rotation-summary-name", default=None)
    parser.add_argument(
        "--rotation-labels-csv",
        type=Path,
        help="Manual filename,correct_rotation labels; overrides TEDS-based selection",
    )
    parser.add_argument("--min-teds-gain", type=float, default=0.01)
    parser.add_argument("--num-ocr-shards", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy: bool) -> str:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return "copy"
    os.symlink(src.resolve(), dst)
    return "symlink"


def resolve_rotation_summary(rotation_run_dir: Path, name: str | None) -> Path:
    if name:
        return rotation_run_dir / name
    for candidate in ("rotation_summary_gt_canon.json", "rotation_summary.json"):
        path = rotation_run_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No rotation summary found in {rotation_run_dir}; expected "
        "rotation_summary_gt_canon.json or rotation_summary.json"
    )


def write_ocr_shards(output_dir: Path, filenames: list[str], num_shards: int) -> None:
    shard_dir = output_dir / "ocr_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    summary = {"samples": len(filenames), "num_shards": num_shards, "shards": []}
    for idx in range(num_shards):
        start = (idx * len(filenames)) // num_shards
        end = ((idx + 1) * len(filenames)) // num_shards
        shard_names = filenames[start:end]
        subset_path = shard_dir / f"subset_{idx:02d}.txt"
        subset_path.write_text("\n".join(shard_names) + "\n", encoding="utf-8")
        summary["shards"].append(
            {"index": idx, "samples": len(shard_names), "subset": str(subset_path)}
        )
    (shard_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    out = args.output_dir
    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    if img_dir.exists() or img_dir.is_symlink():
        if img_dir.is_symlink() or img_dir.is_file():
            img_dir.unlink()
        else:
            shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    aux = json.loads((args.base_run_dir / "aux.json").read_text(encoding="utf-8"))
    base_manifest_path = args.base_run_dir / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_manifest_by_filename = {row["filename"]: row for row in base_manifest}

    rotation_summary_path: Path | None = None
    rotation_by_base: dict[str, dict] = {}
    if args.rotation_labels_csv:
        with args.rotation_labels_csv.open(newline="", encoding="utf-8") as handle:
            label_rows = list(csv.DictReader(handle))
        seen = set()
        for row in label_rows:
            filename = row["filename"].strip()
            if filename in seen:
                raise ValueError(f"Duplicate rotation label: {filename}")
            seen.add(filename)
            rotation = int(row["correct_rotation"])
            if rotation not in (0, 90, 270):
                raise ValueError(f"Invalid rotation {rotation} for {filename}")
            if filename not in base_manifest_by_filename:
                raise ValueError(f"Rotation label not present in base manifest: {filename}")
            if rotation:
                rotation_by_base[filename] = {
                    "base_filename": filename,
                    "best_rotation": rotation,
                    "selection_source": "manual_visual_label",
                }
    else:
        rotation_summary_path = resolve_rotation_summary(
            args.rotation_run_dir, args.rotation_summary_name
        )
        rotation_summary = json.loads(rotation_summary_path.read_text(encoding="utf-8"))
        for row in rotation_summary["rows"]:
            if row["best_rotation"] != 0 and row["delta_best_vs_rot0_teds"] > args.min_teds_gain:
                rotation_by_base[row["base_filename"]] = row

    applied = []
    for filename in sorted(aux):
        base_src = args.base_run_dir / "images" / filename
        source = base_src
        rotation = 0
        rotation_variant = None
        if filename in rotation_by_base:
            row = rotation_by_base[filename]
            rotation = int(row["best_rotation"])
            rotation_variant = f"{Path(filename).stem}__rot{rotation}.png"
            source = args.rotation_run_dir / "images" / rotation_variant
            if not source.exists():
                raise FileNotFoundError(source)
        elif not base_src.exists():
            raise FileNotFoundError(base_src)

        mode = link_or_copy(source, img_dir / filename, args.copy_images)
        base_row = base_manifest_by_filename.get(filename, {})
        applied.append(
            {
                "filename": filename,
                "applied_rotation": rotation,
                "rotation_variant": rotation_variant,
                "image_source": str(source),
                "image_mode": mode,
                "paper_id": base_row.get("paper_id"),
                "gt_html": base_row.get("gt_html"),
                "gt_xml": base_row.get("gt_xml"),
                "original_table_png": base_row.get("table_png"),
            }
        )

    (out / "aux.json").write_text(json.dumps(aux, ensure_ascii=False), encoding="utf-8")
    (out / "subset.txt").write_text("\n".join(sorted(aux)) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(applied, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_ocr_shards(out, sorted(aux), args.num_ocr_shards)
    summary = {
        "base_run_dir": str(args.base_run_dir),
        "rotation_run_dir": str(args.rotation_run_dir),
        "rotation_summary": str(rotation_summary_path) if rotation_summary_path else None,
        "rotation_labels_csv": str(args.rotation_labels_csv) if args.rotation_labels_csv else None,
        "output_dir": str(out),
        "samples": len(aux),
        "rotated_samples": sum(1 for row in applied if row["applied_rotation"] != 0),
        "min_teds_gain": args.min_teds_gain,
        "num_ocr_shards": args.num_ocr_shards,
        "images_dir": str(img_dir),
        "purpose": "Full updated TargetDomain paper collection with fixed rotations before PSENet+MASTER.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
