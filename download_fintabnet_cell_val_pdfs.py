"""Download path-accurate FinTabNet v1.0.0 cell_val PDFs from Kaggle.

The FinTabNet JSONL stores paths such as ``FDX/2018/page_65.pdf``. Keeping the
relative path is important because many different source documents contain the
same basename, for example ``page_65.pdf``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_JSONL = BASE_DIR / "data" / "fintabnet_kaggle" / "FinTabNet_1.0.0_cell_val.jsonl"
DEFAULT_PDF_DIR = BASE_DIR / "data" / "fintabnet_kaggle" / "pdfs"
DEFAULT_TMP_DIR = Path("/tmp") / "fintabnet_kaggle_downloads"
KAGGLE_BIN = BASE_DIR / ".venv" / "bin" / "kaggle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--tmp-dir", type=Path, default=DEFAULT_TMP_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def unique_filenames(jsonl: Path) -> list[str]:
    seen: set[str] = set()
    filenames: list[str] = []
    for line in jsonl.open(encoding="utf-8"):
        filename = json.loads(line)["filename"]
        if filename in seen:
            continue
        seen.add(filename)
        filenames.append(filename)
    return filenames


def main() -> None:
    args = parse_args()
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    filenames = unique_filenames(args.jsonl)
    selected = filenames[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]

    missing = [name for name in selected if not (args.pdf_dir / name).exists()]
    print(f"unique_pdfs_total={len(filenames)}")
    print(f"selected={len(selected)}")
    print(f"missing={len(missing)}")

    for index, rel_path in enumerate(missing, start=1):
        kaggle_file = f"fintabnet/pdf/{rel_path}"
        destination = args.pdf_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(missing)}] {kaggle_file}", flush=True)
        subprocess.run(
            [
                str(KAGGLE_BIN),
                "datasets",
                "download",
                "-d",
                "jiongjiong/fintabnet",
                "-f",
                kaggle_file,
                "-p",
                str(args.tmp_dir),
                "--force",
                "--quiet",
            ],
            check=True,
        )
        downloaded = args.tmp_dir / Path(rel_path).name
        if not downloaded.exists():
            raise FileNotFoundError(downloaded)
        destination.write_bytes(downloaded.read_bytes())

    remaining = [name for name in filenames if not (args.pdf_dir / name).exists()]
    print(f"remaining_missing={len(remaining)}")


if __name__ == "__main__":
    main()
