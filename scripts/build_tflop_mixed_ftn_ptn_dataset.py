#!/usr/bin/env python3
"""Build a mixed FinTabNet/PubTabNet TFLOP training dataset.

This is used for rehearsal-style fine-tuning: keep most batches FinTabNet, but
mix in a smaller PubTabNet subset so the decoder does not forget the PTN-style
HTML/output distribution.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
import shutil
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
TFLOP_ROOT = BASE_DIR / "repo" / "TFLOP"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(TFLOP_ROOT))

from dataset.preprocess_data import (  # noqa: E402
    format_pubtabnet_gold_coords,
    preprocess_det_bbox,
)
from dataset.preprocess_data_utils import convert_html_to_otsl  # noqa: E402


DEFAULT_FTN_DATA_DIR = BASE_DIR / "results" / "tflop_fintabnet_train_full_for_longrun"
DEFAULT_PTN_JSONL = BASE_DIR / "data" / "TFLOP-dataset" / "meta_data" / "PubTabNet_2.0.0.jsonl"
DEFAULT_PTN_AMBIGUOUS = (
    BASE_DIR / "data" / "TFLOP-dataset" / "meta_data" / "erroneous_pubtabnet_data.json"
)
DEFAULT_PTN_IMAGE_ROOT = BASE_DIR / "data" / "pubtabnet_hf" / "extracted" / "pubtabnet"
DEFAULT_PTN_PSE_ROOT = BASE_DIR / "data" / "TFLOP-dataset" / "pse_results"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "tflop_mixed_ftn16k_ptn4k_for_10k"


def load_otsl_map() -> dict[str, str]:
    config_path = TFLOP_ROOT / "dataset" / "data_preprocessing_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))["OTSL_TAG"]


def symlink_image(src: Path, dst: Path) -> None:
    src = src.resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def read_jsonl(path: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    rng.shuffle(rows)
    return rows[:limit]


def copy_ftn_entries(
    source_data_dir: Path,
    split: str,
    limit: int,
    out_image_dir: Path,
    rng: random.Random,
) -> list[dict[str, Any]]:
    path = source_data_dir / "meta_data" / f"dataset_{split}.jsonl"
    source_image_dir = source_data_dir / "images" / split
    rows = read_jsonl(path, limit, rng)
    entries: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        old_name = row["file_name"]
        new_name = f"ftn_{idx:06d}_{old_name}"
        symlink_image(source_image_dir / old_name, out_image_dir / new_name)
        row = dict(row)
        row["file_name"] = new_name
        row["source_dataset"] = "fintabnet"
        entries.append(row)
    return entries


def load_pubtabnet_candidates(
    jsonl_path: Path,
    split: str,
    ambiguous_path: Path,
    image_dir: Path,
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    ambiguous: set[str] = set()
    if ambiguous_path.exists():
        for values in json.loads(ambiguous_path.read_text(encoding="utf-8")).values():
            ambiguous.update(values)

    raw_split = "val" if split == "validation" else split
    candidates: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != raw_split:
                continue
            if row["filename"] in ambiguous:
                continue
            if not (image_dir / row["filename"]).exists():
                continue
            candidates.append(row)

    rng.shuffle(candidates)
    return candidates[:limit]


def load_pse_for_selected(pse_paths: list[Path], selected_filenames: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in pse_paths:
        with path.open("rb") as f:
            rows = pickle.load(f)
        for row in rows:
            fn = row["file_name"]
            if fn in selected_filenames:
                out[fn] = row
    missing = selected_filenames - set(out)
    if missing:
        examples = ", ".join(sorted(list(missing))[:5])
        raise RuntimeError(f"Missing PSE detections for {len(missing)} selected PTN files, e.g. {examples}")
    return out


def build_ptn_entries(
    split: str,
    limit: int,
    out_image_dir: Path,
    rng: random.Random,
    ptn_jsonl: Path,
    ptn_ambiguous: Path,
    ptn_image_root: Path,
    ptn_pse_root: Path,
    otsl_map: dict[str, str],
) -> list[dict[str, Any]]:
    image_split = "val" if split == "validation" else split
    image_dir = ptn_image_root / image_split
    selected = load_pubtabnet_candidates(
        ptn_jsonl, split, ptn_ambiguous, image_dir, limit, rng
    )
    selected_names = {row["filename"] for row in selected}

    if split == "train":
        pse_paths = sorted((ptn_pse_root / "train").glob("detection_results_*.pkl"))
    else:
        pse_paths = [ptn_pse_root / "val" / "detection_results_0.pkl"]
    pse_by_name = load_pse_for_selected(pse_paths, selected_names)

    entries: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        filename = row["filename"]
        new_name = f"ptn_{idx:06d}_{filename}"
        symlink_image(image_dir / filename, out_image_dir / new_name)

        otsl_seq, num_rows, num_cols = convert_html_to_otsl(
            html_seq=row["html"]["structure"]["tokens"],
            otsl_tag_maps=otsl_map,
        )
        gold_bbox_seq = format_pubtabnet_gold_coords(row["html"]["cells"])
        dr_coord = preprocess_det_bbox(
            pse_by_name[filename]["bbox"],
            row["html"]["cells"],
            IOU_threshold=0.1,
            IOP_threshold=0.1,
        )
        if not dr_coord:
            continue
        entries.append(
            {
                "file_name": new_name,
                "dr_coord": dr_coord,
                "gold_coord": gold_bbox_seq,
                "org_html": row["html"]["structure"]["tokens"],
                "otsl_seq": otsl_seq,
                "num_rows": num_rows,
                "num_cols": num_cols,
                "split": split,
                "source_dataset": "pubtabnet",
                "source_filename": filename,
            }
        )
    if len(entries) < limit:
        raise RuntimeError(f"Only built {len(entries)} PTN {split} entries, expected {limit}")
    return entries[:limit]


def write_jsonl(path: Path, entries: list[dict[str, Any]], split: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            row = dict(entry)
            row["split"] = split
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ftn-data-dir", type=Path, default=DEFAULT_FTN_DATA_DIR)
    parser.add_argument("--ptn-jsonl", type=Path, default=DEFAULT_PTN_JSONL)
    parser.add_argument("--ptn-ambiguous", type=Path, default=DEFAULT_PTN_AMBIGUOUS)
    parser.add_argument("--ptn-image-root", type=Path, default=DEFAULT_PTN_IMAGE_ROOT)
    parser.add_argument("--ptn-pse-root", type=Path, default=DEFAULT_PTN_PSE_ROOT)
    parser.add_argument("--ftn-train-samples", type=int, default=16000)
    parser.add_argument("--ptn-train-samples", type=int, default=4000)
    parser.add_argument("--ftn-val-samples", type=int, default=800)
    parser.add_argument("--ptn-val-samples", type=int, default=200)
    parser.add_argument("--bbox-token-cnt", type=int, default=640)
    parser.add_argument("--max-length", type=int, default=1376)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.force:
        shutil.rmtree(args.out_dir)

    meta_dir = args.out_dir / "meta_data"
    train_img_dir = args.out_dir / "images" / "train"
    val_img_dir = args.out_dir / "images" / "validation"
    meta_dir.mkdir(parents=True, exist_ok=True)
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    otsl_map = load_otsl_map()
    rng = random.Random(args.seed)

    ftn_train = copy_ftn_entries(args.ftn_data_dir, "train", args.ftn_train_samples, train_img_dir, rng)
    ptn_train = build_ptn_entries(
        "train",
        args.ptn_train_samples,
        train_img_dir,
        rng,
        args.ptn_jsonl,
        args.ptn_ambiguous,
        args.ptn_image_root,
        args.ptn_pse_root,
        otsl_map,
    )
    ftn_val = copy_ftn_entries(args.ftn_data_dir, "validation", args.ftn_val_samples, val_img_dir, rng)
    ptn_val = build_ptn_entries(
        "validation",
        args.ptn_val_samples,
        val_img_dir,
        rng,
        args.ptn_jsonl,
        args.ptn_ambiguous,
        args.ptn_image_root,
        args.ptn_pse_root,
        otsl_map,
    )

    train_entries = ftn_train + ptn_train
    val_entries = ftn_val + ptn_val
    rng.shuffle(train_entries)
    rng.shuffle(val_entries)

    write_jsonl(meta_dir / "dataset_train.jsonl", train_entries, "train")
    write_jsonl(meta_dir / "dataset_validation.jsonl", val_entries, "validation")

    data_config_path = args.out_dir / "data_config.yaml"
    data_config_path.write_text(
        "\n".join(
            [
                f"image_path: {args.out_dir / 'images'}",
                f"meta_data_path: {meta_dir}",
                "input_size:",
                "  height: 768",
                "  width: 768",
                "window_size: 8",
                "align_along_axis: False",
                f"max_length: {args.max_length}",
                f"bbox_token_cnt: {args.bbox_token_cnt}",
                "use_cell_bbox: False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "train_entries": len(train_entries),
        "validation_entries": len(val_entries),
        "train_mix": {"fintabnet": len(ftn_train), "pubtabnet": len(ptn_train)},
        "validation_mix": {"fintabnet": len(ftn_val), "pubtabnet": len(ptn_val)},
        "seed": args.seed,
        "data_config": str(data_config_path),
        "bbox_token_cnt": args.bbox_token_cnt,
        "max_length": args.max_length,
        "strategy": "mixed FTN+PTN rehearsal fine-tuning dataset",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
