"""Merge TFLOP full_model_inference.json shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="shard_*/full_model_inference.json")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged: dict[str, Any] = {}
    shard_summaries = []
    for inference_path in sorted(args.shard_dir.glob(args.pattern)):
        data = json.loads(inference_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"{inference_path} is not a dict JSON")
        overlap = sorted(set(merged) & set(data))
        if overlap:
            raise ValueError(f"Duplicate inference keys in {inference_path}: {overlap[:5]}")
        merged.update(data)
        shard_summaries.append({"path": str(inference_path), "samples": len(data)})

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    summary = {
        "samples": len(merged),
        "output_json": str(args.output_json),
        "shards": shard_summaries,
    }
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
