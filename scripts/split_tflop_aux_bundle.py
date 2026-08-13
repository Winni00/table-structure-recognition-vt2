"""Split a TFLOP aux bundle into deterministic shards."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aux-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--aux-rec-pkl", type=Path, default=None)
    parser.add_argument("--prefix", default="shard")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def shard_ranges(n_items: int, n_shards: int) -> list[tuple[int, int]]:
    if n_shards <= 0:
        raise ValueError("num-shards must be positive")
    ranges = []
    for idx in range(n_shards):
        start = (idx * n_items) // n_shards
        end = ((idx + 1) * n_items) // n_shards
        ranges.append((start, end))
    return ranges


def main() -> None:
    args = parse_args()
    aux: dict[str, Any] = json.loads(args.aux_json.read_text(encoding="utf-8"))
    filenames = sorted(aux)

    aux_rec = None
    if args.aux_rec_pkl:
        with args.aux_rec_pkl.open("rb") as f:
            aux_rec = pickle.load(f)
        missing = sorted(set(filenames) - set(aux_rec))
        if missing:
            raise ValueError(f"aux_rec is missing {len(missing)} keys, first: {missing[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "samples": len(filenames),
        "num_shards": args.num_shards,
        "aux_json": str(args.aux_json),
        "aux_rec_pkl": str(args.aux_rec_pkl) if args.aux_rec_pkl else None,
        "shards": [],
    }
    for idx, (start, end) in enumerate(shard_ranges(len(filenames), args.num_shards)):
        shard_names = filenames[start:end]
        shard_name = f"{args.prefix}_{idx:02d}"
        subset_path = args.output_dir / f"{shard_name}.txt"
        aux_path = args.output_dir / f"{shard_name}_aux.json"
        rec_path = args.output_dir / f"{shard_name}_aux_rec.pkl"

        subset_path.write_text("\n".join(shard_names) + "\n", encoding="utf-8")
        aux_path.write_text(
            json.dumps({name: aux[name] for name in shard_names}, ensure_ascii=False),
            encoding="utf-8",
        )

        rec_regions = None
        if aux_rec is not None:
            shard_rec = {name: aux_rec[name] for name in shard_names}
            with rec_path.open("wb") as f:
                pickle.dump(shard_rec, f)
            rec_regions = sum(len(v) for v in shard_rec.values())

        summary["shards"].append(
            {
                "index": idx,
                "samples": len(shard_names),
                "subset": str(subset_path),
                "aux_json": str(aux_path),
                "aux_rec_pkl": str(rec_path) if aux_rec is not None else None,
                "regions": rec_regions,
            }
        )

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
