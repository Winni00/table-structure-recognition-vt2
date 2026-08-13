"""Fetch missing PubTabNet v2 validation images from the public full archive.

The local TFLOP bundle contains all PubTabNet validation metadata but only 8958
of the 9115 validation images. For an exact val/dev benchmark run we download
the public ``ajimeno/PubTabNet`` archive and extract only the missing PNGs.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
PUBTABNET_JSONL = BASE_DIR / "data" / "TFLOP-dataset" / "meta_data" / "PubTabNet_2.0.0.jsonl"
VALIDATION_DIR = BASE_DIR / "data" / "TFLOP-dataset" / "images" / "validation"
ARCHIVE_DIR = BASE_DIR / "data" / "pubtabnet_original"


def missing_validation_filenames() -> list[str]:
    missing: list[str] = []
    with PUBTABNET_JSONL.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != "val":
                continue
            filename = row["filename"]
            if not (VALIDATION_DIR / filename).exists():
                missing.append(filename)
    return missing


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    missing = missing_validation_filenames()
    print(f"missing_before={len(missing)}")
    if not missing:
        return

    archive_path = Path(
        hf_hub_download(
            repo_id="ajimeno/PubTabNet",
            repo_type="dataset",
            filename="pubtabnet.tar.gz",
            local_dir=ARCHIVE_DIR,
            local_dir_use_symlinks=False,
        )
    )
    print(f"archive={archive_path}")

    wanted = set(missing)
    extracted = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            basename = Path(member.name).name
            if basename not in wanted:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            (VALIDATION_DIR / basename).write_bytes(source.read())
            extracted += 1
            print(f"[{extracted}/{len(wanted)}] {basename}", flush=True)

    missing_after = missing_validation_filenames()
    print(f"extracted={extracted}")
    print(f"missing_after={len(missing_after)}")
    if missing_after:
        print("still_missing:")
        for filename in missing_after:
            print(filename)


if __name__ == "__main__":
    main()
