#!/usr/bin/env python3
"""Split a non-empty line-oriented text file into balanced shards."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--prefix", default="subset")
    args = parser.parse_args()
    lines = [line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No lines in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = [[] for _ in range(args.num_shards)]
    for index, line in enumerate(lines):
        shards[index % args.num_shards].append(line)
    for index, shard in enumerate(shards):
        path = args.output_dir / f"{args.prefix}_{index:02d}.txt"
        path.write_text("\n".join(shard) + "\n", encoding="utf-8")
    print(f"split {len(lines)} lines into {[len(shard) for shard in shards]}")


if __name__ == "__main__":
    main()
