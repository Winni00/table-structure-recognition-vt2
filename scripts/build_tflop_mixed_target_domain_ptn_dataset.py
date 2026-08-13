#!/usr/bin/env python3
"""Build TargetDomain + PubTabNet replay datasets for conservative TFLOP fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from build_tflop_mixed_ftn_ptn_dataset import (
    DEFAULT_PTN_AMBIGUOUS,
    DEFAULT_PTN_IMAGE_ROOT,
    DEFAULT_PTN_JSONL,
    DEFAULT_PTN_PSE_ROOT,
    build_ptn_entries,
    load_otsl_map,
    symlink_image,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_DOMAIN_DIR = ROOT / "results" / "tflop_target_domain_train_pseudo_labels"
DEFAULT_OUTPUT = ROOT / "results" / "tflop_mixed_target_domain15k_ptn5k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target_domain-data-dir", type=Path, default=DEFAULT_TARGET_DOMAIN_DIR)
    parser.add_argument("--ptn-jsonl", type=Path, default=DEFAULT_PTN_JSONL)
    parser.add_argument("--ptn-ambiguous", type=Path, default=DEFAULT_PTN_AMBIGUOUS)
    parser.add_argument("--ptn-image-root", type=Path, default=DEFAULT_PTN_IMAGE_ROOT)
    parser.add_argument("--ptn-pse-root", type=Path, default=DEFAULT_PTN_PSE_ROOT)
    parser.add_argument("--target_domain-train-samples", type=int, default=15000)
    parser.add_argument("--ptn-train-samples", type=int, default=5000)
    parser.add_argument("--target_domain-val-samples", type=int, default=800)
    parser.add_argument("--ptn-val-samples", type=int, default=200)
    parser.add_argument("--bbox-token-cnt", type=int, default=640)
    parser.add_argument("--max-length", type=int, default=1376)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy_target_domain_entries(
    source_dir: Path,
    split: str,
    limit: int,
    output_images: Path,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows = load_rows(source_dir / "meta_data" / f"dataset_{split}.jsonl")
    if len(rows) < limit:
        raise RuntimeError(f"TargetDomain {split} has {len(rows)} accepted entries, requested {limit}")
    rng.shuffle(rows)
    entries: list[dict[str, Any]] = []
    for index, source in enumerate(rows[:limit]):
        row = dict(source)
        old_name = row["file_name"]
        new_name = f"target_domain_{index:06d}_{old_name}"
        symlink_image(source_dir / "images" / split / old_name, output_images / new_name)
        row["file_name"] = new_name
        row["source_dataset"] = "target_domain_paper_collection"
        entries.append(row)
    return entries


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and args.force:
        shutil.rmtree(args.out_dir)
    meta_dir = args.out_dir / "meta_data"
    train_images = args.out_dir / "images" / "train"
    validation_images = args.out_dir / "images" / "validation"
    for path in (meta_dir, train_images, validation_images):
        path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    otsl_map = load_otsl_map()
    target_domain_train = copy_target_domain_entries(
        args.target_domain_data_dir, "train", args.target_domain_train_samples, train_images, rng
    )
    ptn_train = build_ptn_entries(
        "train",
        args.ptn_train_samples,
        train_images,
        rng,
        args.ptn_jsonl,
        args.ptn_ambiguous,
        args.ptn_image_root,
        args.ptn_pse_root,
        otsl_map,
    )
    target_domain_validation = copy_target_domain_entries(
        args.target_domain_data_dir,
        "validation",
        args.target_domain_val_samples,
        validation_images,
        rng,
    )
    ptn_validation = build_ptn_entries(
        "validation",
        args.ptn_val_samples,
        validation_images,
        rng,
        args.ptn_jsonl,
        args.ptn_ambiguous,
        args.ptn_image_root,
        args.ptn_pse_root,
        otsl_map,
    )

    train = target_domain_train + ptn_train
    validation = target_domain_validation + ptn_validation
    rng.shuffle(train)
    rng.shuffle(validation)
    write_jsonl(meta_dir / "dataset_train.jsonl", train, "train")
    write_jsonl(meta_dir / "dataset_validation.jsonl", validation, "validation")

    data_config = args.out_dir / "data_config.yaml"
    data_config.write_text(
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
        "seed": args.seed,
        "train": {"target_domain": len(target_domain_train), "pubtabnet": len(ptn_train)},
        "validation": {
            "target_domain": len(target_domain_validation),
            "pubtabnet": len(ptn_validation),
        },
        "data_config": str(data_config),
        "strategy": "TargetDomain adaptation with PubTabNet experience replay",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
