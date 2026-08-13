"""Copy PNG images while setting DPI metadata.

This does not rescale pixels. It writes a PNG pHYs/DPI metadata tag so the
output files explicitly report the requested DPI. This is useful for documenting
the TableFormer paper preprocessing requirement without mutating original data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=72)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.png"))
    if args.limit is not None:
        files = files[: args.limit]

    samples_before = []
    samples_after = []
    for idx, src in enumerate(files):
        dst = args.output_dir / src.name
        with Image.open(src) as image:
            if idx < 10:
                samples_before.append(
                    {
                        "filename": src.name,
                        "size_px": list(image.size),
                        "dpi": image.info.get("dpi"),
                        "info_keys": sorted(image.info.keys()),
                    }
                )
            image.save(dst, dpi=(args.dpi, args.dpi))
        if idx < 10:
            with Image.open(dst) as converted:
                samples_after.append(
                    {
                        "filename": dst.name,
                        "size_px": list(converted.size),
                        "dpi": converted.info.get("dpi"),
                        "info_keys": sorted(converted.info.keys()),
                    }
                )

    payload = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "requested_dpi": args.dpi,
        "converted": len(files),
        "samples_before": samples_before,
        "samples_after": samples_after,
    }
    manifest = args.manifest or (args.output_dir / "dpi_manifest.json")
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
