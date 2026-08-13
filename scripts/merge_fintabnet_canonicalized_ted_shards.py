#!/usr/bin/env python3
"""Merge FTN canonicalized TEDS shard files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RUN_DIR = BASE_DIR / "results" / "tflop_fintabnet_full_annotation_excl_problem_pages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-name", default="ted_score_output_ftn_canonicalized_sharded.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    missing = []
    for shard_idx in range(args.num_shards):
        path = (
            args.run_dir
            / f"ted_score_output_ftn_canonicalized_shard_{shard_idx}_of_{args.num_shards}.json"
        )
        if not path.exists():
            missing.append(str(path))
            continue
        rows.extend(json.loads(path.read_text(encoding="utf-8")))

    if missing:
        raise SystemExit("Missing shard files:\n" + "\n".join(missing))

    rows.sort(key=lambda row: row[0])
    out_path = args.run_dir / args.output_name
    out_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    summary = {
        "output": str(out_path),
        "num_shards": args.num_shards,
        "num_samples": len(rows),
        "canonicalization": "flatten thead/tbody/tfoot so all tr nodes are direct table children",
        "teds_s": sum(row[-2] for row in rows) / len(rows),
        "teds": sum(row[-1] for row in rows) / len(rows),
        "teds_s_1": sum(row[-2] == 1.0 for row in rows),
        "teds_1": sum(row[-1] == 1.0 for row in rows),
    }
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
