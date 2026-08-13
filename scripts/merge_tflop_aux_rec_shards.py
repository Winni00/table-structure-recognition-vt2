"""Merge TFLOP OCR aux_rec pickle shards."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="aux_rec_*.pkl")
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = {}
    shard_summaries = []
    for pkl_path in sorted(args.shard_dir.glob(args.pattern)):
        with pkl_path.open("rb") as f:
            shard = pickle.load(f)
        overlap = sorted(set(merged) & set(shard))
        if overlap:
            raise ValueError(f"Duplicate keys in {pkl_path}: {overlap[:5]}")
        merged.update(shard)
        shard_summaries.append(
            {
                "path": str(pkl_path),
                "samples": len(shard),
                "regions": sum(len(v) for v in shard.values()),
            }
        )

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as f:
        pickle.dump(merged, f)

    summary = {
        "samples": len(merged),
        "regions": sum(len(v) for v in merged.values()),
        "output_pkl": str(args.output_pkl),
        "shards": shard_summaries,
    }
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
