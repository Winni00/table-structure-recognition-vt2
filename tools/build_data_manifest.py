#!/usr/bin/env python3
"""Create a deterministic SHA-256 manifest for an external data archive."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Dataset or archive directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.resolve() != output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as manifest:
        manifest.write("sha256\tsize_bytes\trelative_path\n")
        for path in files:
            relative = path.relative_to(root).as_posix()
            manifest.write(f"{sha256(path)}\t{path.stat().st_size}\t{relative}\n")

    print(f"Wrote {len(files)} entries to {output}")


if __name__ == "__main__":
    main()
