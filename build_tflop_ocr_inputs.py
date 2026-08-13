"""Build TFLOP OCR inputs for the selected hard table crops.

This script runs the local Tesseract CLI on each hard-table PNG and writes one
OCR JSON file per table in the format expected by `tflop_inference.py`:

    <output-dir>/<table_id>.ocr.json

Each OCR JSON contains a list of entries:

[
  {"bbox": [x1, y1, x2, y2], "text": "recognized text"},
  ...
]

The implementation uses Tesseract TSV output so that we keep word-level boxes
and text, which is what TFLOP needs for its pointer-based decoding path.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_HARD_TABLES_JSON = (
    BASE_DIR / "data" / "paper_collection_meta" / "selected_hard_tables.json"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "paper_collection_meta" / "tflop_ocr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Tesseract on the 12 hard table crops and export TFLOP-style OCR "
            "JSON files with bbox + text entries."
        )
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_HARD_TABLES_JSON,
        help="Path to the selected hard tables metadata JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where one <table_id>.ocr.json file per table will be written.",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract language code, for example eng.",
    )
    return parser.parse_args()


def ensure_tesseract() -> str:
    """Ensure the Tesseract CLI is available."""
    tesseract_bin = shutil.which("tesseract")
    if tesseract_bin is None:
        raise FileNotFoundError("Tesseract binary not found in PATH.")
    return tesseract_bin


def run_tesseract_tsv(tesseract_bin: str, image_path: Path, lang: str) -> list[dict]:
    """Run Tesseract TSV output on one image and return word-level OCR boxes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpbase = Path(tmpdir) / "ocr"
        cmd = [
            tesseract_bin,
            str(image_path),
            str(tmpbase),
            "-l",
            lang,
            "--psm",
            "6",
            "tsv",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tsv_path = tmpbase.with_suffix(".tsv")

        entries = []
        with tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("text") or "").strip()
                if not text:
                    continue

                try:
                    conf = float(row.get("conf", "-1"))
                except ValueError:
                    conf = -1.0
                if conf < 0:
                    continue

                left = int(row["left"])
                top = int(row["top"])
                width = int(row["width"])
                height = int(row["height"])
                entries.append(
                    {
                        "bbox": [left, top, left + width, top + height],
                        "text": text,
                        "conf": conf,
                    }
                )
        return entries


def main() -> None:
    args = parse_args()
    tesseract_bin = ensure_tesseract()
    rows = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for row in rows:
        table_id = row["table_id"]
        image_path = Path(row["input_png"])
        ocr_entries = run_tesseract_tsv(tesseract_bin, image_path, args.lang)

        output_path = args.output_dir / f"{table_id}.ocr.json"
        output_path.write_text(json.dumps(ocr_entries, indent=2), encoding="utf-8")

        summary.append(
            {
                "table_id": table_id,
                "image_path": str(image_path),
                "ocr_output": str(output_path),
                "ocr_entries": len(ocr_entries),
            }
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("TFLOP OCR input generation finished.")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
