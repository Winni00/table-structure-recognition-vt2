#!/usr/bin/env python3
"""Run the PSENet + MASTER OCR stage used by the VT2 experiments."""

import argparse
import json
import os
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLEMASTER_DIR = REPOSITORY_ROOT / "external/TableMASTER-mmocr"
DEFAULT_CHECKPOINT_DIR = REPOSITORY_ROOT / "models/tablemaster_mmocr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect table text regions with PSENet, recognise them with MASTER, "
            "and write the per-image OCR records as a pickle file."
        )
    )
    parser.add_argument(
        "--subset",
        type=Path,
        required=True,
        help="UTF-8 text file containing one image filename per line.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing the images named by --subset.",
    )
    parser.add_argument(
        "--tablemaster-dir",
        type=Path,
        default=DEFAULT_TABLEMASTER_DIR,
        help="TableMASTER-mmocr checkout (default: external/TableMASTER-mmocr).",
    )
    parser.add_argument(
        "--pse-config",
        type=Path,
        help=(
            "PSENet config (default: <tablemaster-dir>/configs/textdet/psenet/"
            "psenet_r50_fpnf_600e_pubtabnet.py)."
        ),
    )
    parser.add_argument(
        "--pse-checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR / "pse_epoch_600.pth",
        help="PSENet checkpoint (default: models/tablemaster_mmocr/pse_epoch_600.pth).",
    )
    parser.add_argument(
        "--master-config",
        type=Path,
        help=(
            "MASTER config (default: <tablemaster-dir>/configs/textrecog/master/"
            "master_lmdb_ResnetExtra_tableRec_dataset_dynamic_mmfp16.py)."
        ),
    )
    parser.add_argument(
        "--master-checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR / "master_epoch_6.pth",
        help="MASTER checkpoint (default: models/tablemaster_mmocr/master_epoch_6.pth).",
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        required=True,
        help="Destination pickle; a sibling .summary.json file is also written.",
    )
    parser.add_argument(
        "--on-error",
        choices=("fail", "empty"),
        default="fail",
        help="Fail immediately or emit an empty list for an image-level error.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{label} does not exist or is not a file: {path}")
    return path


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def patched_master_config(config_path: Path, tablemaster_dir: Path, output_dir: Path) -> Path:
    """Copy the MASTER config and make its import-time file paths absolute."""
    text = config_path.read_text(encoding="utf-8")
    base_path = tablemaster_dir / "configs/_base_/default_runtime.py"
    alphabet_path = tablemaster_dir / "tools/data/alphabet/textline_recognition_alphabet.txt"
    if not alphabet_path.is_file():
        alphabet_path = (
            tablemaster_dir
            / "table_recognition/demo/alphabet/textline_recognition_alphabet.txt"
        )
    require_file(base_path, "TableMASTER default runtime config")
    require_file(alphabet_path, "MASTER alphabet")

    original_base = "_base_ = [\n    '../../_base_/default_runtime.py',\n]"
    original_alphabet = (
        "alphabet_file = './tools/data/alphabet/textline_recognition_alphabet.txt'"
    )
    if original_base not in text or original_alphabet not in text:
        raise SystemExit(
            "MASTER config does not contain the expected base/alphabet declarations; "
            "refusing to patch an unknown config variant."
        )
    text = text.replace(original_base, f"_base_ = [\n    '{base_path}',\n]", 1)
    text = text.replace(
        original_alphabet, f"alphabet_file = '{alphabet_path}'", 1
    )

    patched_path = output_dir / "master_config_absolute_paths.py"
    patched_path.write_text(text, encoding="utf-8")
    return patched_path


def main() -> None:
    args = parse_args()
    tablemaster_dir = args.tablemaster_dir.expanduser().resolve()
    if not tablemaster_dir.is_dir():
        raise SystemExit(f"TableMASTER directory does not exist: {tablemaster_dir}")

    pse_config = args.pse_config or (
        tablemaster_dir
        / "configs/textdet/psenet/psenet_r50_fpnf_600e_pubtabnet.py"
    )
    master_config = args.master_config or (
        tablemaster_dir
        / "configs/textrecog/master/"
        "master_lmdb_ResnetExtra_tableRec_dataset_dynamic_mmfp16.py"
    )
    subset = require_file(args.subset, "Subset file")
    images_dir = args.images_dir.expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"Image directory does not exist: {images_dir}")
    pse_config = require_file(pse_config, "PSENet config")
    pse_checkpoint = require_file(args.pse_checkpoint, "PSENet checkpoint")
    master_config = require_file(master_config, "MASTER config")
    master_checkpoint = require_file(args.master_checkpoint, "MASTER checkpoint")
    output_pkl = args.output_pkl.expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    filenames = [
        line.strip()
        for line in subset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing_images = [name for name in filenames if not (images_dir / name).is_file()]
    if missing_images:
        preview = ", ".join(missing_images[:3])
        raise SystemExit(
            f"{len(missing_images)} subset image(s) are missing from {images_dir}: {preview}"
        )

    patched_config = patched_master_config(
        master_config, tablemaster_dir, output_pkl.parent
    )
    sys.path.insert(0, str(tablemaster_dir))
    from table_recognition.table_inference import (  # pylint: disable=import-outside-toplevel
        Detect_Inference,
        End2End,
        Recognition_Inference,
    )
    from tqdm import tqdm  # pylint: disable=import-outside-toplevel

    with working_directory(tablemaster_dir):
        detector = Detect_Inference(str(pse_config), str(pse_checkpoint))
        recognizer = Recognition_Inference(str(patched_config), str(master_checkpoint))
        end2end = End2End(detector, recognizer)

    outputs = {}
    errors = {}
    for filename in tqdm(filenames, desc="TableMASTER PSENet+MASTER"):
        try:
            with working_directory(tablemaster_dir):
                result, _ = end2end.predict(str(images_dir / filename))
        except Exception as exc:  # Keep optional long-running shard jobs alive.
            if args.on_error == "fail":
                raise
            result = []
            errors[filename] = repr(exc)
        outputs[filename] = result

    with output_pkl.open("wb") as handle:
        pickle.dump(outputs, handle)

    summary = {
        "samples": len(outputs),
        "total_items": int(sum(len(items) for items in outputs.values())),
        "output_pkl": str(output_pkl),
        "pse_config": str(pse_config),
        "pse_checkpoint": str(pse_checkpoint),
        "master_config": str(master_config),
        "master_checkpoint": str(master_checkpoint),
        "sample_counts": {name: len(items) for name, items in outputs.items()},
        "errors": errors,
        "num_errors": len(errors),
    }
    summary_path = output_pkl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
