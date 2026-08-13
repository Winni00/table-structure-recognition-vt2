"""Create a TFLOP auxiliary bundle from a raw recognizer field.

This is used to remove hidden recognizer fallbacks from an existing bundle.
For example, `--text-field pymupdf_text` keeps raw PyMuPDF text and leaves
empty PyMuPDF regions empty instead of replacing them with MASTER output.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-field", required=True)
    parser.add_argument("--source-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    for name in ("aux.json", "subset.txt", "manifest.json"):
        shutil.copy2(source / name, output / name)
    images = output / "images"
    if images.is_symlink() or images.is_file():
        images.unlink()
    elif images.exists():
        shutil.rmtree(images)
    images.symlink_to((source / "images").resolve())

    with (source / "aux_rec.pkl").open("rb") as handle:
        bundle = pickle.load(handle)

    total = 0
    empty = 0
    output_bundle = {}
    for filename, regions in bundle.items():
        output_regions = []
        for region in regions:
            total += 1
            raw_text = str(region.get(args.text_field, "") or "").strip()
            if not raw_text:
                empty += 1
            output_region = dict(region)
            output_region["text"] = raw_text
            output_region["active_text_source"] = args.source_name
            output_region["fallback_used"] = False
            output_regions.append(output_region)
        output_bundle[filename] = output_regions

    with (output / "aux_rec.pkl").open("wb") as handle:
        pickle.dump(output_bundle, handle)

    summary = {
        "source_dir": str(source),
        "output_dir": str(output),
        "text_field": args.text_field,
        "active_text_source": args.source_name,
        "samples": len(output_bundle),
        "regions": total,
        "empty_regions": empty,
        "empty_ratio": empty / max(1, total),
        "fallback_used": False,
    }
    (output / "pure_text_bundle_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
