#!/usr/bin/env python3
"""Merge sharded TFLOP TEDS outputs into ted_score_output.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--output-name",
        default="ted_score_output_sharded.json",
        help="Merged output filename inside run-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = []
    missing = []
    for shard_index in range(args.num_shards):
        path = (
            args.run_dir
            / f"ted_score_output_shard_{shard_index:02d}_of_{args.num_shards:02d}.json"
        )
        if not path.exists():
            missing.append(str(path))
            continue
        merged.extend(json.loads(path.read_text(encoding="utf-8")))

    if missing:
        raise SystemExit("Missing shard outputs:\n" + "\n".join(missing))

    merged.sort(key=lambda row: row[0])
    out_path = args.run_dir / args.output_name
    out_path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")

    teds_s = sum(row[-2] for row in merged) / len(merged) if merged else 0.0
    teds = sum(row[-1] for row in merged) / len(merged) if merged else 0.0
    summary = {
        "run_dir": str(args.run_dir),
        "output": str(out_path),
        "num_shards": args.num_shards,
        "num_samples": len(merged),
        "teds_s": teds_s,
        "teds": teds,
        "teds_s_1": sum(row[-2] == 1.0 for row in merged),
        "teds_1": sum(row[-1] == 1.0 for row in merged),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
