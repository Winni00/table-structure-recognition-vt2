#!/usr/bin/env python3
"""Evaluate TFLOP TEDS for one shard of a full_model_inference.json file."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from pathlib import Path

from tqdm import tqdm


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
TFLOP_REPO = BASE_DIR / "repo" / "TFLOP"
sys.path.insert(0, str(TFLOP_REPO))

from evaluate_ted import evaluate_distance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--legacy-strip-cell-contents", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    inference_path = args.run_dir / "full_model_inference.json"
    model_inference = json.loads(inference_path.read_text(encoding="utf-8"))
    data_collection = [
        (
            key,
            value["pred_string"],
            value["answer_string"],
            args.legacy_strip_cell_contents,
        )
        for key, value in sorted(model_inference.items(), key=lambda item: item[0])
    ]
    shard_items = [
        item
        for index, item in enumerate(data_collection)
        if index % args.num_shards == args.shard_index
    ]

    results = []
    for start in tqdm(
        range(0, len(shard_items), args.batch_size),
        desc=f"TEDS shard {args.shard_index}/{args.num_shards}",
    ):
        batch = shard_items[start : start + args.batch_size]
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            results.extend(pool.map(evaluate_distance, batch))

    shard_path = (
        args.run_dir
        / f"ted_score_output_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    )
    shard_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    teds_s = sum(row[-2] for row in results) / len(results) if results else 0.0
    teds = sum(row[-1] for row in results) / len(results) if results else 0.0
    summary = {
        "run_dir": str(args.run_dir),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_samples": len(results),
        "teds_s": teds_s,
        "teds": teds,
        "teds_s_1": sum(row[-2] == 1.0 for row in results),
        "teds_1": sum(row[-1] == 1.0 for row in results),
    }
    summary_path = shard_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
