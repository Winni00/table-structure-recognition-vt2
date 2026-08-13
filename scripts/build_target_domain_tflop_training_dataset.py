#!/usr/bin/env python3
"""Build high-confidence TFLOP training labels from TargetDomain HTML and OCR regions.

TargetDomain data provides table crops and GT HTML but no cell bounding boxes. This
adapter monotonically aligns PSENet+MASTER text regions to GT cells. Only
high-confidence tables are retained; uncertain pseudo-labels are reported and
excluded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from rapidfuzz.fuzz import ratio


ROOT = Path(__file__).resolve().parents[1]
TFLOP_ROOT = ROOT / "repo" / "TFLOP"
DEFAULT_INVENTORY = ROOT / "results" / "target_domain_train_inventory_split"
DEFAULT_OCR = ROOT / "results" / "target_domain_train_psenet_master" / "aux_rec.pkl"
DEFAULT_OUTPUT = ROOT / "results" / "tflop_target_domain_train_pseudo_labels"


@dataclass(frozen=True)
class Cell:
    index: int
    text: str


@dataclass(frozen=True)
class Region:
    index: int
    bbox: list[float]
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-dir", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--ocr-pkl", type=Path, default=DEFAULT_OCR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cell-similarity", type=float, default=0.72)
    parser.add_argument("--min-character-coverage", type=float, default=0.78)
    parser.add_argument("--min-mean-similarity", type=float, default=0.82)
    parser.add_argument("--max-regions-per-cell", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=400)
    parser.add_argument("--max-regions", type=int, default=639)
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def load_convert_html_to_otsl():
    module_path = TFLOP_ROOT / "dataset" / "preprocess_data_utils.py"
    spec = importlib.util.spec_from_file_location("tflop_preprocess_data_utils_target_domain", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.convert_html_to_otsl


def load_otsl_map() -> dict[str, str]:
    path = TFLOP_ROOT / "dataset" / "data_preprocessing_config.json"
    return json.loads(path.read_text(encoding="utf-8"))["OTSL_TAG"]


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    return "".join(char for char in value if char.isalnum())


def plain_text(tag: Tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()


def direct_rows(section: Tag) -> list[Tag]:
    return [node for node in section.find_all("tr") if node.find_parent(["thead", "tbody", "tfoot", "table"]) is section]


def table_structure_and_cells(html_path: Path) -> tuple[list[str], list[Cell]]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    table = soup.find("table")
    if table is None:
        raise ValueError("GT HTML has no table")

    sections: list[tuple[str, list[Tag]]] = []
    thead = table.find("thead", recursive=False)
    tbody_nodes = table.find_all("tbody", recursive=False)
    direct = [node for node in table.find_all("tr", recursive=False)]
    sections.append(("thead", direct_rows(thead) if thead else []))
    body_rows: list[Tag] = []
    for tbody in tbody_nodes:
        body_rows.extend(direct_rows(tbody))
    body_rows.extend(direct)
    sections.append(("tbody", body_rows))

    tokens: list[str] = []
    cells: list[Cell] = []
    for section_name, rows in sections:
        tokens.append(f"<{section_name}>")
        for row in rows:
            tokens.append("<tr>")
            row_cells = row.find_all(["td", "th"], recursive=False)
            for tag in row_cells:
                attributes: list[str] = []
                for attr in ("colspan", "rowspan"):
                    raw = str(tag.get(attr, "1"))
                    try:
                        span = max(1, int(raw))
                    except ValueError:
                        span = 1
                    if span > 1:
                        attributes.append(f' {attr}="{span}"')
                if attributes:
                    tokens.append("<td")
                    tokens.extend(attributes)
                    tokens.append(">")
                else:
                    tokens.append("<td>")
                tokens.append("</td>")
                cells.append(Cell(len(cells), plain_text(tag)))
            tokens.append("</tr>")
        tokens.append(f"</{section_name}>")
    return tokens, cells


def unique_name(row: dict[str, Any]) -> str:
    return f"{row['paper_id']}__{Path(row['table_png']).name}"


def regions_from_ocr(items: list[dict[str, Any]], max_regions: int) -> list[Region]:
    regions: list[Region] = []
    for index, item in enumerate(items[:max_regions]):
        bbox = [float(value) for value in item["bbox"]]
        if len(bbox) != 4:
            continue
        text = str(item.get("text", "")).strip()
        regions.append(Region(index, bbox, text))
    return regions


def segment_similarity(cell_text: str, regions: list[Region]) -> float:
    target = normalize_text(cell_text)
    observed = normalize_text(" ".join(region.text for region in regions))
    if not target or not observed:
        return 0.0
    return ratio(target, observed) / 100.0


def align_cells_regions(
    cells: list[Cell], regions: list[Region], max_regions_per_cell: int
) -> tuple[dict[int, tuple[list[Region], float]], int]:
    nonempty = [cell for cell in cells if normalize_text(cell.text)]
    n, m = len(nonempty), len(regions)
    neg_inf = -math.inf
    scores = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, str, int, float] | None]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]
    scores[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            base = scores[i][j]
            if base == neg_inf:
                continue
            if j < m and base - 0.28 > scores[i][j + 1]:
                scores[i][j + 1] = base - 0.28
                back[i][j + 1] = (i, j, "skip_region", 1, 0.0)
            if i < n:
                if base - 0.85 > scores[i + 1][j]:
                    scores[i + 1][j] = base - 0.85
                    back[i + 1][j] = (i, j, "skip_cell", 0, 0.0)
                for count in range(1, min(max_regions_per_cell, m - j) + 1):
                    sim = segment_similarity(nonempty[i].text, regions[j : j + count])
                    candidate = base + (2.0 * sim - 0.72) - 0.025 * (count - 1)
                    if candidate > scores[i + 1][j + count]:
                        scores[i + 1][j + count] = candidate
                        back[i + 1][j + count] = (i, j, "map", count, sim)

    mapping: dict[int, tuple[list[Region], float]] = {}
    skipped_regions = 0
    i, j = n, m
    while i or j:
        step = back[i][j]
        if step is None:
            break
        prev_i, prev_j, action, count, sim = step
        if action == "map":
            mapping[nonempty[prev_i].index] = (regions[prev_j : prev_j + count], sim)
        elif action == "skip_region":
            skipped_regions += 1
        i, j = prev_i, prev_j
    return mapping, skipped_regions


def union_bbox(regions: list[Region]) -> list[float]:
    return [
        min(region.bbox[0] for region in regions),
        min(region.bbox[1] for region in regions),
        max(region.bbox[2] for region in regions),
        max(region.bbox[3] for region in regions),
    ]


def gold_coord_string(text: str, regions: list[Region] | None) -> str:
    if not text or not regions:
        return "-1.0 -1.0 -1.0 -1.0 1 "
    bbox = union_bbox(regions)
    return " ".join([*(f"{value:.2f}" for value in bbox), "2", text])


def build_entry(
    row: dict[str, Any],
    ocr: dict[str, list[dict[str, Any]]],
    convert_html_to_otsl,
    otsl_map: dict[str, str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    name = unique_name(row)
    report: dict[str, Any] = {"filename": name, "paper_id": row["paper_id"], "split": row["split"]}
    if name not in ocr:
        return None, {**report, "accepted": False, "reason": "missing_ocr"}

    try:
        structure, cells = table_structure_and_cells(ROOT / row["table_html"])
        otsl_seq, num_rows, num_cols = convert_html_to_otsl(structure, otsl_map)
    except Exception as exc:  # noqa: BLE001
        return None, {**report, "accepted": False, "reason": "html_or_otsl", "detail": str(exc)}
    if not cells or len(cells) > args.max_cells:
        return None, {**report, "accepted": False, "reason": "cell_count", "cells": len(cells)}

    regions = regions_from_ocr(ocr[name], args.max_regions)
    if not regions:
        return None, {**report, "accepted": False, "reason": "no_regions"}
    mapping, skipped_regions = align_cells_regions(cells, regions, args.max_regions_per_cell)

    accepted_mapping = {
        cell_index: (mapped_regions, sim)
        for cell_index, (mapped_regions, sim) in mapping.items()
        if sim >= args.cell_similarity
    }
    nonempty = [cell for cell in cells if normalize_text(cell.text)]
    total_chars = sum(len(normalize_text(cell.text)) for cell in nonempty)
    mapped_chars = sum(
        len(normalize_text(cells[index].text)) for index in accepted_mapping
    )
    character_coverage = mapped_chars / max(total_chars, 1)
    weighted_similarity = sum(
        len(normalize_text(cells[index].text)) * sim
        for index, (_mapped_regions, sim) in accepted_mapping.items()
    ) / max(mapped_chars, 1)

    report.update(
        {
            "cells": len(cells),
            "nonempty_cells": len(nonempty),
            "regions": len(regions),
            "mapped_cells": len(accepted_mapping),
            "character_coverage": character_coverage,
            "weighted_similarity": weighted_similarity,
            "skipped_regions": skipped_regions,
        }
    )
    if character_coverage < args.min_character_coverage:
        return None, {**report, "accepted": False, "reason": "low_coverage"}
    if weighted_similarity < args.min_mean_similarity:
        return None, {**report, "accepted": False, "reason": "low_similarity"}

    ordered = sorted(
        accepted_mapping.items(),
        key=lambda item: min(region.index for region in item[1][0]),
    )
    dr_coord: dict[int, list[Any]] = {}
    for group_index, (cell_index, (mapped_regions, _sim)) in enumerate(ordered):
        dr_coord[group_index] = [
            [region.bbox for region in mapped_regions],
            cell_index,
            cells[cell_index].text,
        ]
    gold_coord = [
        gold_coord_string(
            cell.text,
            accepted_mapping[cell.index][0] if cell.index in accepted_mapping else None,
        )
        for cell in cells
    ]
    entry = {
        "file_name": name,
        "dr_coord": dr_coord,
        "gold_coord": gold_coord,
        "org_html": structure,
        "otsl_seq": otsl_seq,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "split": row["split"],
        "source_filename": row["article_pdf"],
        "table_id": row["table_id"],
        "source_dataset": "target_domain_paper_collection",
    }
    return entry, {**report, "accepted": True, "reason": "accepted"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_root = args.output_dir / "images"
    image_root.mkdir(exist_ok=True)
    meta_dir = args.output_dir / "meta_data"
    meta_dir.mkdir(exist_ok=True)

    with args.ocr_pkl.open("rb") as handle:
        ocr = pickle.load(handle)
    convert_html_to_otsl = load_convert_html_to_otsl()
    otsl_map = load_otsl_map()

    all_reports: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for split in ("train", "validation"):
        image_dir = image_root / split
        image_dir.mkdir(exist_ok=True)
        source_rows = load_jsonl(args.inventory_dir / f"{split}.jsonl")
        entries: list[dict[str, Any]] = []
        for row in source_rows:
            entry, report = build_entry(row, ocr, convert_html_to_otsl, otsl_map, args)
            all_reports.append(report)
            if entry is None:
                continue
            entries.append(entry)
            source = ROOT / row["table_png"]
            target = image_dir / entry["file_name"]
            if not target.exists():
                if args.copy_images:
                    shutil.copy2(source, target)
                else:
                    os.symlink(source.resolve(), target)
        write_jsonl(meta_dir / f"dataset_{split}.jsonl", entries)
        summaries[split] = {"seen": len(source_rows), "accepted": len(entries)}

    (args.output_dir / "data_config.yaml").write_text(
        "\n".join(
            [
                f"image_path: {image_root}",
                f"meta_data_path: {meta_dir}",
                "input_size:",
                "  height: 768",
                "  width: 768",
                "window_size: 8",
                "align_along_axis: False",
                "max_length: 1376",
                "bbox_token_cnt: 640",
                "use_cell_bbox: False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    write_jsonl(args.output_dir / "alignment_report.jsonl", all_reports)
    accepted_reports = [row for row in all_reports if row["accepted"]]
    summary = {
        "inventory_dir": str(args.inventory_dir),
        "ocr_pkl": str(args.ocr_pkl),
        "thresholds": {
            "cell_similarity": args.cell_similarity,
            "min_character_coverage": args.min_character_coverage,
            "min_mean_similarity": args.min_mean_similarity,
        },
        "splits": summaries,
        "reports": len(all_reports),
        "accepted": len(accepted_reports),
        "mean_character_coverage": sum(row["character_coverage"] for row in accepted_reports)
        / max(len(accepted_reports), 1),
        "mean_weighted_similarity": sum(row["weighted_similarity"] for row in accepted_reports)
        / max(len(accepted_reports), 1),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
