#!/usr/bin/env python3
"""Write a compact summary for a TFLOP ted_score_output.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit(f"No TEDS rows in {args.input}")
    summary = {
        "samples": len(rows),
        "num_samples": len(rows),
        "teds_s": sum(float(row[-2]) for row in rows) / len(rows),
        "teds": sum(float(row[-1]) for row in rows) / len(rows),
        "teds_s_1": sum(float(row[-2]) == 1.0 for row in rows),
        "teds_1": sum(float(row[-1]) == 1.0 for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
