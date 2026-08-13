#!/usr/bin/env python3
"""Inventory TargetDomain table data and create document-disjoint train/validation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lxml import html
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "target_domain_train"
DEFAULT_TEST_MANIFEST = (
    ROOT / "results" / "tflop_paper_collection_updated_crops_ocr_style_public" / "manifest.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "target_domain_train_inventory_split"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()


def load_test_papers(path: Path) -> set[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["paper_id"]) for row in rows}


def load_coordinates(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report malformed source data
        return {}, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return {}, "coordinate JSON is not an object"
    return value, None


def inspect_html(path: Path) -> tuple[dict[str, int], str | None]:
    try:
        document = html.fromstring(path.read_bytes())
        rows = document.xpath(".//tr")
        cells = document.xpath(".//td|.//th")
        text_cells = sum(bool(" ".join(cell.itertext()).strip()) for cell in cells)
        return {"html_rows": len(rows), "html_cells": len(cells), "html_text_cells": text_cells}, None
    except Exception as exc:  # noqa: BLE001
        return {"html_rows": 0, "html_cells": 0, "html_text_cells": 0}, f"{type(exc).__name__}: {exc}"


def inspect_image(path: Path) -> tuple[dict[str, int], str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return {"image_width": image.width, "image_height": image.height}, None
    except Exception as exc:  # noqa: BLE001
        return {"image_width": 0, "image_height": 0}, f"{type(exc).__name__}: {exc}"


def page_image_for(directory: Path, table_stem: str) -> Path | None:
    candidates = sorted(directory.glob(f"page_*_{table_stem}.png"))
    return candidates[0] if candidates else None


def inventory(data_dir: Path, test_papers: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for directory in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        paper_id = directory.name
        coordinate_path = directory / "table_coord.json"
        coordinates, coordinate_error = load_coordinates(coordinate_path) if coordinate_path.exists() else ({}, "missing")
        if coordinate_error:
            errors.append({"paper_id": paper_id, "table": "", "kind": "coordinates", "detail": coordinate_error})

        article_pdf = directory / f"{paper_id}.pdf"
        article_xml = directory / f"{paper_id}.xml"

        for html_path in sorted(directory.glob("*_table_*.html")):
            stem = html_path.stem
            table_png = directory / f"{stem}.png"
            table_xml = directory / f"{stem}.xml"
            page_png = page_image_for(directory, stem)
            coord = coordinates.get(stem)

            html_stats, html_error = inspect_html(html_path)
            image_stats, image_error = inspect_image(table_png) if table_png.exists() else (
                {"image_width": 0, "image_height": 0},
                "missing",
            )

            missing = [
                name
                for name, path in (
                    ("pdf", article_pdf),
                    ("article_xml", article_xml),
                    ("table_png", table_png),
                    ("table_xml", table_xml),
                )
                if not path.exists()
            ]
            if page_png is None:
                missing.append("page_png")
            if coord is None:
                missing.append("table_coord")
            if html_error:
                errors.append({"paper_id": paper_id, "table": stem, "kind": "html", "detail": html_error})
            if image_error:
                errors.append({"paper_id": paper_id, "table": stem, "kind": "image", "detail": image_error})

            valid_coord = isinstance(coord, dict) and all(
                key in coord for key in ("left", "top", "right", "bottom", "page_no")
            )
            record = {
                "paper_id": paper_id,
                "table_id": stem,
                "table_html": str(html_path.relative_to(ROOT)),
                "table_xml": str(table_xml.relative_to(ROOT)),
                "table_png": str(table_png.relative_to(ROOT)),
                "page_png": str(page_png.relative_to(ROOT)) if page_png else None,
                "article_pdf": str(article_pdf.relative_to(ROOT)),
                "article_xml": str(article_xml.relative_to(ROOT)),
                "table_coord": coord,
                "coord_valid": valid_coord,
                "overlaps_fixed_test": paper_id in test_papers,
                "missing": missing,
                "valid_for_split": not missing and not html_error and not image_error and valid_coord,
                **html_stats,
                **image_stats,
            }
            records.append(record)

    return records, errors


def split_documents(
    records: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["valid_for_split"] and not row["overlaps_fixed_test"]:
            by_paper[row["paper_id"]].append(row)

    ranked = sorted(
        by_paper,
        key=lambda paper: hashlib.sha256(f"{seed}:{paper}".encode()).hexdigest(),
    )
    target_tables = round(sum(len(rows) for rows in by_paper.values()) * validation_fraction)
    validation: set[str] = set()
    validation_tables = 0
    for paper in ranked:
        if validation_tables >= target_tables:
            break
        validation.add(paper)
        validation_tables += len(by_paper[paper])
    return set(by_paper) - validation, validation


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_papers = load_test_papers(args.test_manifest)
    records, errors = inventory(args.data_dir, test_papers)
    train_papers, validation_papers = split_documents(records, args.validation_fraction, args.seed)

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test_overlap": [], "invalid": []}
    for row in records:
        paper = row["paper_id"]
        if row["overlaps_fixed_test"]:
            split = "test_overlap"
        elif not row["valid_for_split"]:
            split = "invalid"
        elif paper in validation_papers:
            split = "validation"
        elif paper in train_papers:
            split = "train"
        else:
            split = "invalid"
        enriched = {**row, "split": split}
        split_rows[split].append(enriched)

    for split, rows in split_rows.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    write_jsonl(args.output_dir / "all_tables.jsonl", [row for rows in split_rows.values() for row in rows])
    write_jsonl(args.output_dir / "errors.jsonl", errors)

    summary = {
        "data_dir": str(args.data_dir),
        "test_manifest": str(args.test_manifest),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "tables_total": len(records),
        "papers_total": len({row["paper_id"] for row in records}),
        "tables_by_split": {key: len(value) for key, value in split_rows.items()},
        "papers_by_split": {
            key: len({row["paper_id"] for row in value}) for key, value in split_rows.items()
        },
        "missing_counts": dict(Counter(item for row in records for item in row["missing"])),
        "errors": len(errors),
        "cell_coordinates_available": False,
        "table_coordinates_available": sum(row["coord_valid"] for row in records),
        "fixed_test_papers": len(test_papers),
        "train_validation_paper_overlap": len(train_papers & validation_papers),
        "train_fixed_test_paper_overlap": len(train_papers & test_papers),
        "validation_fixed_test_paper_overlap": len(validation_papers & test_papers),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
