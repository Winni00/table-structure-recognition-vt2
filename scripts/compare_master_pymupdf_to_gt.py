"""Compare MASTER and PyMuPDF text directly against Paper Collection GT HTML.

Both text sources use the same PSENet regions. Region strings are concatenated
in their existing reading order and compared with the ordered text from the GT
table. A compact metric without whitespace is included because PSENet regions
often split one GT cell into several fragments.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import pickle
import re
import shutil
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Any

from bs4 import BeautifulSoup
from rapidfuzz.distance import Levenshtein


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_MASTER = ROOT / "results/tflop_paper_collection_updated_crops_broadbestrot_master_public"
DEFAULT_PYMUPDF = ROOT / "results/tflop_paper_collection_updated_crops_broadbestrot_pymupdf_public"
DEFAULT_VERIFY = ROOT / "results/pymupdf_extraction_verification"
DEFAULT_OUT = ROOT / "results/paper_collection_master_pymupdf_gt_text_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-run", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--pymupdf-run", type=Path, default=DEFAULT_PYMUPDF)
    parser.add_argument("--verification-dir", type=Path, default=DEFAULT_VERIFY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--examples-per-group", type=int, default=10)
    return parser.parse_args()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\u00ad", "").replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", normalize_text(value)).casefold()


def gt_text_from_html(value: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(value or "", "lxml")
    cells = [normalize_text(cell.get_text(" ", strip=True)) for cell in soup.select("th,td")]
    cells = [cell for cell in cells if cell]
    return normalize_text(" ".join(cells)), cells


def plain_text_fragment(value: str) -> str:
    """Remove recognizer formatting tags while preserving their text content."""
    value = str(value or "")
    if "<" in value and ">" in value:
        value = BeautifulSoup(value, "lxml").get_text(" ", strip=True)
    return normalize_text(value)


def source_text(regions: list[dict[str, Any]], field: str) -> str:
    return normalize_text(" ".join(plain_text_fragment(region.get(field, "")) for region in regions))


def metrics(candidate: str, reference: str) -> dict[str, float | bool]:
    cand = normalize_text(candidate).casefold()
    ref = normalize_text(reference).casefold()
    cand_compact = compact_text(candidate)
    ref_compact = compact_text(reference)
    char_distance = Levenshtein.distance(cand, ref)
    compact_distance = Levenshtein.distance(cand_compact, ref_compact)
    cand_tokens = cand.split()
    ref_tokens = ref.split()
    token_distance = Levenshtein.distance(cand_tokens, ref_tokens)
    overlap = sum((Counter(cand_tokens) & Counter(ref_tokens)).values())
    precision = overlap / max(1, len(cand_tokens))
    recall = overlap / max(1, len(ref_tokens))
    token_f1 = 2 * precision * recall / max(1e-12, precision + recall)
    cand_chars = Counter(cand_compact)
    ref_chars = Counter(ref_compact)
    char_overlap = sum((cand_chars & ref_chars).values())
    char_precision = char_overlap / max(1, len(cand_compact))
    char_recall = char_overlap / max(1, len(ref_compact))
    char_bag_f1 = 2 * char_precision * char_recall / max(1e-12, char_precision + char_recall)
    return {
        "exact": cand == ref,
        "compact_exact": cand_compact == ref_compact,
        "char_similarity": 1.0 - char_distance / max(1, len(ref), len(cand)),
        "compact_char_similarity": 1.0 - compact_distance / max(1, len(ref_compact), len(cand_compact)),
        "char_bag_f1": char_bag_f1,
        "cer": char_distance / max(1, len(ref)),
        "wer": token_distance / max(1, len(ref_tokens)),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": token_f1,
        "extra_token_ratio": max(0, len(cand_tokens) - overlap) / max(1, len(ref_tokens)),
        "missing_token_ratio": max(0, len(ref_tokens) - overlap) / max(1, len(ref_tokens)),
        "length_ratio": len(cand_compact) / max(1, len(ref_compact)),
    }


def load_teds(run_dir: Path) -> dict[str, dict[str, float]]:
    path = run_dir / "gt_canonicalized_rescore/ted_score_output.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row[0]: {"teds_s": float(row[4]), "teds": float(row[5])} for row in rows}


def f(value: float) -> str:
    return f"{value:.4f}"


def excerpt(value: str, limit: int = 700) -> str:
    value = normalize_text(value)
    return value if len(value) <= limit else value[:limit] + " ..."


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    aux = json.loads((args.master_run / "aux.json").read_text(encoding="utf-8"))
    with (args.master_run / "aux_rec.pkl").open("rb") as handle:
        master_regions = pickle.load(handle)
    with (args.pymupdf_run / "aux_rec.pkl").open("rb") as handle:
        pdf_regions = pickle.load(handle)
    master_teds = load_teds(args.master_run)
    pdf_teds = load_teds(args.pymupdf_run)

    rows: list[dict[str, Any]] = []
    for filename in sorted(set(aux) & set(master_regions) & set(pdf_regions)):
        gt_text, gt_cells = gt_text_from_html(aux[filename].get("html", ""))
        master_text = source_text(master_regions[filename], "text")
        # Use the raw PDF extraction. The run's `text` field can contain the
        # MASTER fallback when a mapped PDF region is empty.
        pymupdf_text = source_text(pdf_regions[filename], "pymupdf_text")
        mm = metrics(master_text, gt_text)
        pm = metrics(pymupdf_text, gt_text)
        rows.append({
            "filename": filename,
            "paper_id": filename.split("__", 1)[0],
            "gt_cells": len(gt_cells),
            "pse_regions": len(master_regions[filename]),
            "gt_text": gt_text,
            "master_text": master_text,
            "pymupdf_text": pymupdf_text,
            **{f"master_{k}": v for k, v in mm.items()},
            **{f"pymupdf_{k}": v for k, v in pm.items()},
            "delta_compact_similarity": float(pm["compact_char_similarity"]) - float(mm["compact_char_similarity"]),
            "master_teds_s": master_teds.get(filename, {}).get("teds_s"),
            "master_teds": master_teds.get(filename, {}).get("teds"),
            "pymupdf_master_fallback_teds_s": pdf_teds.get(filename, {}).get("teds_s"),
            "pymupdf_master_fallback_teds": pdf_teds.get(filename, {}).get("teds"),
        })

    json_fields = list(rows[0]) if rows else []
    with (out / "per_table_text_comparison.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "per_table_text_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=json_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "samples": len(rows),
        "comparison_unit": "Ordered table text from identical PSENet regions versus ordered GT cell text",
        "master": {
            "mean_compact_char_similarity": mean(float(r["master_compact_char_similarity"]) for r in rows),
            "median_compact_char_similarity": median(float(r["master_compact_char_similarity"]) for r in rows),
            "mean_token_f1": mean(float(r["master_token_f1"]) for r in rows),
            "mean_token_precision": mean(float(r["master_token_precision"]) for r in rows),
            "mean_token_recall": mean(float(r["master_token_recall"]) for r in rows),
            "mean_extra_token_ratio": mean(float(r["master_extra_token_ratio"]) for r in rows),
            "mean_missing_token_ratio": mean(float(r["master_missing_token_ratio"]) for r in rows),
            "mean_char_bag_f1": mean(float(r["master_char_bag_f1"]) for r in rows),
            "median_length_ratio": median(float(r["master_length_ratio"]) for r in rows),
            "mean_cer": mean(float(r["master_cer"]) for r in rows),
            "mean_wer": mean(float(r["master_wer"]) for r in rows),
            "exact_tables": sum(bool(r["master_exact"]) for r in rows),
            "compact_exact_tables": sum(bool(r["master_compact_exact"]) for r in rows),
        },
        "pymupdf": {
            "mean_compact_char_similarity": mean(float(r["pymupdf_compact_char_similarity"]) for r in rows),
            "median_compact_char_similarity": median(float(r["pymupdf_compact_char_similarity"]) for r in rows),
            "mean_token_f1": mean(float(r["pymupdf_token_f1"]) for r in rows),
            "mean_token_precision": mean(float(r["pymupdf_token_precision"]) for r in rows),
            "mean_token_recall": mean(float(r["pymupdf_token_recall"]) for r in rows),
            "mean_extra_token_ratio": mean(float(r["pymupdf_extra_token_ratio"]) for r in rows),
            "mean_missing_token_ratio": mean(float(r["pymupdf_missing_token_ratio"]) for r in rows),
            "mean_char_bag_f1": mean(float(r["pymupdf_char_bag_f1"]) for r in rows),
            "median_length_ratio": median(float(r["pymupdf_length_ratio"]) for r in rows),
            "mean_cer": mean(float(r["pymupdf_cer"]) for r in rows),
            "mean_wer": mean(float(r["pymupdf_wer"]) for r in rows),
            "exact_tables": sum(bool(r["pymupdf_exact"]) for r in rows),
            "compact_exact_tables": sum(bool(r["pymupdf_compact_exact"]) for r in rows),
        },
        "wins_by_compact_char_similarity": {
            "master": sum(float(r["delta_compact_similarity"]) < -1e-9 for r in rows),
            "pymupdf": sum(float(r["delta_compact_similarity"]) > 1e-9 for r in rows),
            "ties": sum(abs(float(r["delta_compact_similarity"])) <= 1e-9 for r in rows),
        },
        "limitations": [
            "GT has whole cells but PSENet may split one cell into several regions.",
            "Whitespace-insensitive metrics reduce segmentation bias but do not prove exact region-to-cell assignment.",
            "TEDS is reported only as downstream context and is not used to choose the better extracted text.",
            "The available PyMuPDF TFLOP TEDS run falls back to MASTER for empty PDF regions; it is not a pure-PyMuPDF end-to-end run.",
        ],
    }
    score_groups = {
        "good_master_teds_ge_0_9": [r for r in rows if r["master_teds"] is not None and float(r["master_teds"]) >= 0.9],
        "bad_master_teds_lt_0_7": [r for r in rows if r["master_teds"] is not None and float(r["master_teds"]) < 0.7],
    }
    summary["by_downstream_master_teds"] = {}
    for group_name, group_rows in score_groups.items():
        summary["by_downstream_master_teds"][group_name] = {
            "samples": len(group_rows),
            "master_mean_token_f1": mean(float(r["master_token_f1"]) for r in group_rows),
            "pymupdf_mean_token_f1": mean(float(r["pymupdf_token_f1"]) for r in group_rows),
            "master_mean_compact_similarity": mean(float(r["master_compact_char_similarity"]) for r in group_rows),
            "pymupdf_mean_compact_similarity": mean(float(r["pymupdf_compact_char_similarity"]) for r in group_rows),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    n = args.examples_per_group
    groups = {
        "master_best": sorted(rows, key=lambda r: float(r["master_compact_char_similarity"]), reverse=True)[:n],
        "master_worst": sorted(rows, key=lambda r: float(r["master_compact_char_similarity"]))[:n],
        "pymupdf_best": sorted(rows, key=lambda r: float(r["pymupdf_compact_char_similarity"]), reverse=True)[:n],
        "pymupdf_worst": sorted(rows, key=lambda r: float(r["pymupdf_compact_char_similarity"]))[:n],
        "pymupdf_better": sorted(rows, key=lambda r: float(r["delta_compact_similarity"]), reverse=True)[:n],
        "master_better": sorted(rows, key=lambda r: float(r["delta_compact_similarity"]))[:n],
        "good_downstream_examples": sorted(score_groups["good_master_teds_ge_0_9"], key=lambda r: float(r["master_teds"]), reverse=True)[:n],
        "bad_downstream_examples": sorted(score_groups["bad_master_teds_lt_0_7"], key=lambda r: float(r["master_teds"]))[:n],
    }
    (out / "example_manifest.json").write_text(json.dumps(groups, indent=2, ensure_ascii=False), encoding="utf-8")

    visual_index: dict[str, Path] = {}
    verification_summary = args.verification_dir / "summary.json"
    if verification_summary.exists():
        verification_data = json.loads(verification_summary.read_text(encoding="utf-8"))
        for row in verification_data.get("sample_summaries", []):
            if row.get("visual"):
                visual_index[row["filename"]] = ROOT / row["visual"]
    visuals_out = out / "mapping_visuals"
    visuals_out.mkdir(exist_ok=True)
    copied_visuals: dict[str, str] = {}
    for group_rows in groups.values():
        for row in group_rows:
            source = visual_index.get(row["filename"])
            if source and source.exists() and row["filename"] not in copied_visuals:
                target = visuals_out / source.name
                shutil.copy2(source, target)
                copied_visuals[row["filename"]] = str(target.relative_to(out))

    sections = []
    for group_name, group_rows in groups.items():
        cards = []
        for row in group_rows:
            visual = copied_visuals.get(row["filename"])
            visual_html = f'<img src="{html.escape(visual)}" loading="lazy">' if visual else ""
            cards.append(f"""
            <article>
              <h3>{html.escape(row['filename'])}</h3>
              <p><b>MASTER similarity:</b> {f(float(row['master_compact_char_similarity']))} &nbsp;
                 <b>PyMuPDF similarity:</b> {f(float(row['pymupdf_compact_char_similarity']))} &nbsp;
                 <b>Delta PDF-MASTER:</b> {float(row['delta_compact_similarity']):+.4f}</p>
              {visual_html}
              <details><summary>GT / MASTER / PyMuPDF text</summary>
                <p><b>GT:</b> {html.escape(excerpt(row['gt_text']))}</p>
                <p><b>MASTER:</b> {html.escape(excerpt(row['master_text']))}</p>
                <p><b>PyMuPDF:</b> {html.escape(excerpt(row['pymupdf_text']))}</p>
              </details>
            </article>""")
        sections.append(f"<h2>{group_name.replace('_', ' ').title()}</h2>{''.join(cards)}")
    report_html = f"""<!doctype html><html><head><meta charset="utf-8"><title>MASTER vs PyMuPDF vs GT</title>
    <style>body{{font:16px Arial,sans-serif;max-width:1400px;margin:30px auto;padding:0 20px;color:#222}}
    article{{border-top:1px solid #aaa;padding:14px 0}} img{{max-width:100%;max-height:850px;border:1px solid #ddd}}
    p{{line-height:1.45}} summary{{cursor:pointer;font-weight:bold}}</style></head><body>
    <h1>MASTER vs PyMuPDF text compared with GT</h1>
    <p>Direct table-text comparison on the same PSENet regions. Whitespace-insensitive character similarity is the primary ranking metric.</p>
    {''.join(sections)}</body></html>"""
    (out / "examples.html").write_text(report_html, encoding="utf-8")

    readme = f"""# MASTER vs PyMuPDF text compared with GT

## Setup

- Samples: {len(rows)}
- Same PSENet regions for both text sources
- GT text: ordered text from all GT HTML cells
- Primary metric: whitespace-insensitive character similarity
- This measures extraction quality before TFLOP layout assignment
- Direct PyMuPDF metrics use only the raw `pymupdf_text` field, without MASTER fallback

## Results

| Text source | Token precision | Token recall | Token F1 | Character-bag F1 | Extra tokens / GT | Missing tokens / GT |
|---|---:|---:|---:|---:|---:|---:|
| MASTER | {f(summary['master']['mean_token_precision'])} | {f(summary['master']['mean_token_recall'])} | {f(summary['master']['mean_token_f1'])} | {f(summary['master']['mean_char_bag_f1'])} | {f(summary['master']['mean_extra_token_ratio'])} | {f(summary['master']['mean_missing_token_ratio'])} |
| PyMuPDF | {f(summary['pymupdf']['mean_token_precision'])} | {f(summary['pymupdf']['mean_token_recall'])} | {f(summary['pymupdf']['mean_token_f1'])} | {f(summary['pymupdf']['mean_char_bag_f1'])} | {f(summary['pymupdf']['mean_extra_token_ratio'])} | {f(summary['pymupdf']['mean_missing_token_ratio'])} |

Wins by compact character similarity:

- MASTER: {summary['wins_by_compact_char_similarity']['master']}
- PyMuPDF: {summary['wins_by_compact_char_similarity']['pymupdf']}
- Ties: {summary['wins_by_compact_char_similarity']['ties']}

## Good and bad downstream cases

| Group | Samples | MASTER Token F1 | PyMuPDF Token F1 | MASTER compact similarity | PyMuPDF compact similarity |
|---|---:|---:|---:|---:|---:|
| Good: MASTER TEDS >= 0.9 | {summary['by_downstream_master_teds']['good_master_teds_ge_0_9']['samples']} | {f(summary['by_downstream_master_teds']['good_master_teds_ge_0_9']['master_mean_token_f1'])} | {f(summary['by_downstream_master_teds']['good_master_teds_ge_0_9']['pymupdf_mean_token_f1'])} | {f(summary['by_downstream_master_teds']['good_master_teds_ge_0_9']['master_mean_compact_similarity'])} | {f(summary['by_downstream_master_teds']['good_master_teds_ge_0_9']['pymupdf_mean_compact_similarity'])} |
| Bad: MASTER TEDS < 0.7 | {summary['by_downstream_master_teds']['bad_master_teds_lt_0_7']['samples']} | {f(summary['by_downstream_master_teds']['bad_master_teds_lt_0_7']['master_mean_token_f1'])} | {f(summary['by_downstream_master_teds']['bad_master_teds_lt_0_7']['pymupdf_mean_token_f1'])} | {f(summary['by_downstream_master_teds']['bad_master_teds_lt_0_7']['master_mean_compact_similarity'])} | {f(summary['by_downstream_master_teds']['bad_master_teds_lt_0_7']['pymupdf_mean_compact_similarity'])} |

## Important limitation

GT provides whole cell strings, while PSENet can split one cell into several text regions. Therefore the direct comparison is performed on ordered table text, not by naively comparing every PSE box with a whole GT cell. The compact metric removes whitespace introduced by different segmentation. It does not replace a true cell-coordinate mapping.

The existing PyMuPDF TFLOP run uses MASTER as a fallback for empty PDF regions. Its TEDS values therefore describe a PyMuPDF+MASTER-fallback run, while the text metrics above evaluate raw PyMuPDF text only.

## Files

- `per_table_text_comparison.csv/jsonl`: all metrics and texts
- `example_manifest.json`: best, worst and largest source differences
- `examples.html`: readable good/bad examples with available mapping visuals
- `mapping_visuals/`: copied PDF mapping evidence where available
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
