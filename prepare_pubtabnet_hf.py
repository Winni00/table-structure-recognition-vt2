"""Download and inspect the original HuggingFace PubTabNet bundle.

This keeps the original PubTabNet path separate from the TFLOP-dataset bundle.
It writes a manifest with the located JSONL, image directory, split counts,
DPI metadata samples, and 1x1..20x10 table-shape counts.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from PIL import Image

from tableformer_pubtabnet_repro import table_shape_from_tokens


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_OUT = BASE_DIR / "data" / "pubtabnet_hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repo-id", default="ajimeno/PubTabNet")
    parser.add_argument("--archive-name", default="pubtabnet.tar.gz")
    return parser.parse_args()


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extract_complete"
    if marker.exists():
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(extract_dir)
    marker.write_text("ok\n", encoding="utf-8")


def find_jsonl(extract_dir: Path) -> Path:
    candidates = sorted(extract_dir.rglob("*.jsonl"))
    for path in candidates:
        with path.open("r", encoding="utf-8") as fh:
            for _ in range(20):
                line = fh.readline()
                if not line:
                    break
                row = json.loads(line)
                if {"filename", "split", "html"}.issubset(row):
                    return path
    raise FileNotFoundError(f"No PubTabNet-like JSONL found under {extract_dir}")


def find_image_root(extract_dir: Path, filename: str) -> Path:
    matches = sorted(extract_dir.rglob(Path(filename).name))
    if not matches:
        raise FileNotFoundError(f"Could not locate image for {filename} under {extract_dir}")
    match = matches[0]
    filename_parts = Path(filename).parts
    if len(filename_parts) == 1:
        return match.parent
    root = match
    for _ in filename_parts:
        root = root.parent
    return root


def image_path_for(row: dict[str, Any], image_root: Path) -> Path:
    direct = image_root / row["filename"]
    if direct.exists():
        return direct
    matches = sorted(image_root.rglob(Path(row["filename"]).name))
    if matches:
        return matches[0]
    return direct


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=args.archive_name,
            local_dir=args.output_dir,
            local_dir_use_symlinks=False,
        )
    )
    extract_dir = args.output_dir / "extracted"
    extract_archive(archive_path, extract_dir)

    jsonl_path = find_jsonl(extract_dir)
    rows = [json.loads(line) for line in jsonl_path.open("r", encoding="utf-8")]
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row.get("split", "")] = split_counts.get(row.get("split", ""), 0) + 1

    val_rows = [row for row in rows if row.get("split") == "val"]
    if not val_rows:
        raise RuntimeError("No val rows found in PubTabNet JSONL.")
    image_root = find_image_root(extract_dir, val_rows[0]["filename"])

    dpi_samples = []
    for row in (val_rows[:3] + val_rows[len(val_rows) // 2 : len(val_rows) // 2 + 3] + val_rows[-3:]):
        path = image_path_for(row, image_root)
        with Image.open(path) as image:
            dpi_samples.append(
                {
                    "filename": row["filename"],
                    "path": str(path),
                    "size_px": list(image.size),
                    "dpi": image.info.get("dpi"),
                    "info_keys": sorted(image.info.keys()),
                }
            )

    val_shapes = [table_shape_from_tokens(row["html"]["structure"]["tokens"]) for row in val_rows]
    val_filter_count = sum(1 for rows_, cols in val_shapes if 1 <= rows_ <= 20 and 1 <= cols <= 10)

    manifest = {
        "repo_id": args.repo_id,
        "archive_path": str(archive_path),
        "extract_dir": str(extract_dir),
        "jsonl_path": str(jsonl_path),
        "image_root": str(image_root),
        "split_counts": split_counts,
        "val_count": len(val_rows),
        "val_1x1_20x10_count": val_filter_count,
        "dpi_samples": dpi_samples,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
