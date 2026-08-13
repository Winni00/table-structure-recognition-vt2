#!/usr/bin/env python3
"""Create visual alignment diagnostics for PTN-Val, PTN-Test, and FTN inputs."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
OUT_DIR = ROOT / "results" / "text_region_alignment_debug"

PTN_JSONL = ROOT / "data" / "TFLOP-dataset" / "meta_data" / "PubTabNet_2.0.0.jsonl"
PTN_VAL_IMAGES = ROOT / "data" / "TFLOP-dataset" / "images" / "validation"
PTN_VAL_PSE = ROOT / "data" / "TFLOP-dataset" / "pse_results" / "val" / "detection_results_0.pkl"
PTN_VAL_AUX_REC = ROOT / "results" / "tflop_pubtabnet_val_annotations_filtered_8958" / "aux_rec.pkl"

PTN_TEST_IMAGES = ROOT / "data" / "TFLOP-dataset" / "images" / "test"
PTN_TEST_OCR = ROOT / "data" / "TFLOP-dataset" / "pse_results" / "test" / "end2end_results.pkl"

FTN_JSONL = ROOT / "data" / "fintabnet_kaggle" / "FinTabNet_1.0.0_cell_val_excluding_28.jsonl"
FTN_IMAGES = ROOT / "results" / "tflop_fintabnet_full_annotation_excl_problem_pages" / "images"
FTN_AUX_REC = ROOT / "results" / "tflop_fintabnet_full_annotation_excl_problem_pages" / "aux_rec.pkl"


COLORS = {
    "gt": (35, 105, 255),
    "pse": (235, 70, 70),
    "input": (30, 160, 80),
    "text": (0, 0, 0),
    "white": (255, 255, 255),
}


def rich_text(tokens: list[str]) -> str:
    return "".join(tokens or [])


def clean_text(text: str, limit: int = 28) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def load_pubtabnet_val_rows(wanted: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with PTN_JSONL.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != "val":
                continue
            fn = row["filename"]
            if fn in wanted:
                rows[fn] = row
                if len(rows) == len(wanted):
                    break
    return rows


def poly_to_bbox(poly: Any) -> list[float] | None:
    vals = [float(x) for x in list(poly)]
    if len(vals) == 4:
        return vals
    if len(vals) == 8:
        xs = vals[0::2]
        ys = vals[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def draw_boxes(
    image_path: Path,
    layers: list[dict[str, Any]],
    title: str,
    subtitle: str,
    out_path: Path,
) -> dict[str, Any]:
    base = Image.open(image_path).convert("RGB")
    width, height = base.size
    scale = min(1.7, max(1.0, 900 / max(width, 1)))
    canvas_img = base.resize((int(width * scale), int(height * scale)))
    top = 88
    legend_h = 44
    canvas = Image.new("RGB", (canvas_img.width, canvas_img.height + top + legend_h), "white")
    canvas.paste(canvas_img, (0, top))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=COLORS["text"])
    draw.text((10, 30), subtitle, fill=(80, 80, 80))

    x = 10
    for layer in layers:
        color = COLORS[layer["color"]]
        draw.rectangle([x, 58, x + 18, 76], outline=color, width=3)
        draw.text((x + 24, 58), f'{layer["name"]}: {len(layer["boxes"])}', fill=COLORS["text"])
        x += 210

    summary: dict[str, Any] = {"image": str(image_path), "size": [width, height], "layers": {}}
    for layer in layers:
        color = COLORS[layer["color"]]
        boxes = []
        for item in layer["boxes"]:
            bbox = item["bbox"]
            if not bbox:
                continue
            x0, y0, x1, y1 = [float(v) * scale for v in bbox]
            y0 += top
            y1 += top
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            label = clean_text(item.get("text", ""))
            if label and layer.get("label", False):
                lx, ly = x0, max(top, y0 - 11)
                draw.rectangle([lx, ly, lx + min(180, 6 * len(label) + 8), ly + 11], fill=COLORS["white"])
                draw.text((lx + 2, ly), label, fill=color)
            boxes.append({"bbox": [round(float(v), 2) for v in bbox], "text": item.get("text", "")})
        summary["layers"][layer["name"]] = {"count": len(boxes), "boxes_preview": boxes[:8]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    summary["output"] = str(out_path)
    return summary


def build_ptn_val_plots() -> list[dict[str, Any]]:
    with PTN_VAL_AUX_REC.open("rb") as f:
        aux_rec: dict[str, list[dict[str, Any]]] = pickle.load(f)
    with PTN_VAL_PSE.open("rb") as f:
        pse_list = pickle.load(f)
    pse_by_name = {entry["file_name"]: entry for entry in pse_list}

    samples = [
        "PMC2871264_002_00.png",
        "PMC3725841_011_00.png",
        "PMC3598671_009_01.png",
        "PMC4176503_005_00.png",
    ]
    samples = [s for s in samples if s in aux_rec and (PTN_VAL_IMAGES / s).exists()]
    rows = load_pubtabnet_val_rows(set(samples))
    summaries = []
    for fn in samples:
        row = rows[fn]
        gt_boxes = []
        gt_nonempty_with_bbox = 0
        for cell in row["html"]["cells"]:
            bbox = cell.get("bbox")
            text = rich_text(cell.get("tokens", []))
            if bbox and len(bbox) == 4:
                gt_boxes.append({"bbox": bbox, "text": text})
                if text:
                    gt_nonempty_with_bbox += 1
        input_boxes = [{"bbox": item["bbox"], "text": item["text"]} for item in aux_rec[fn]]
        pse_entry = pse_by_name.get(fn, {})
        pse_boxes = []
        for i, poly in enumerate(pse_entry.get("bbox", [])):
            bbox = poly_to_bbox(poly)
            if bbox:
                score = pse_entry.get("score", [])
                text = f'{float(score[i]):.2f}' if i < len(score) else ""
                pse_boxes.append({"bbox": bbox, "text": text})
        summary = draw_boxes(
            PTN_VAL_IMAGES / fn,
            [
                {"name": "GT cell boxes", "boxes": gt_boxes, "color": "gt", "label": False},
                {"name": "PSE det boxes", "boxes": pse_boxes, "color": "pse", "label": False},
                {"name": "our TFLOP input", "boxes": input_boxes, "color": "input", "label": True},
            ],
            f"PTN-Val alignment: {fn}",
            "Blue=PubTabNet GT cell boxes. Red=TFLOP PSE detection boxes. Green=our annotation-derived TFLOP input boxes/text.",
            OUT_DIR / "ptn_val" / f"{Path(fn).stem}_alignment.png",
        )
        summary["question_answer"] = {
            "gt_cell_boxes_text": "Original PubTabNet cell annotations, including annotated cells with bbox and their token text.",
            "annotation_derived_tflop_input": "Adapter output: non-empty PubTabNet bbox/text cells serialized as TFLOP OCR-style inputs with score=1.0.",
            "difference": "For PTN-Val they should use the same coordinates/text for non-empty annotated cells; empty cells/no-text cells are omitted from TFLOP input.",
            "gt_boxes_with_bbox": len(gt_boxes),
            "gt_nonempty_with_bbox": gt_nonempty_with_bbox,
            "input_boxes": len(input_boxes),
            "pse_boxes": len(pse_boxes),
        }
        summaries.append(summary)
    return summaries


def build_ptn_test_plots() -> list[dict[str, Any]]:
    with PTN_TEST_OCR.open("rb") as f:
        ocr: dict[str, list[dict[str, Any]]] = pickle.load(f)
    samples = sorted(ocr.keys())[:4]
    summaries = []
    for fn in samples:
        boxes = [{"bbox": poly_to_bbox(item["bbox"]), "text": item.get("text", "")} for item in ocr[fn]]
        boxes = [b for b in boxes if b["bbox"]]
        summary = draw_boxes(
            PTN_TEST_IMAGES / fn,
            [{"name": "OCR input boxes", "boxes": boxes, "color": "input", "label": True}],
            f"PTN-Test OCR input: {fn}",
            "Green=OCR inference boxes/text from end2end_results.pkl. These are inputs to TFLOP, not TFLOP-predicted boxes.",
            OUT_DIR / "ptn_test" / f"{Path(fn).stem}_ocr_input.png",
        )
        summary["question_answer"] = {
            "box_source": "TFLOP bundle end2end_results.pkl",
            "meaning": "Precomputed OCR inference results: bbox, bbox_score, recognized text, recognition score.",
        }
        summaries.append(summary)
    return summaries


def build_ftn_plots() -> list[dict[str, Any]]:
    with FTN_AUX_REC.open("rb") as f:
        aux_rec: dict[str, list[dict[str, Any]]] = pickle.load(f)
    samples = sorted(aux_rec.keys())[:4]
    summaries = []
    for fn in samples:
        input_boxes = [{"bbox": item["bbox"], "text": item["text"]} for item in aux_rec[fn]]
        summary = draw_boxes(
            FTN_IMAGES / fn,
            [{"name": "FTN TFLOP input", "boxes": input_boxes, "color": "input", "label": True}],
            f"FTN crop alignment: {fn}",
            "Green=FinTabNet annotation-derived boxes after PDF-to-image, table-crop offset, and scaling transformation.",
            OUT_DIR / "ftn" / f"{Path(fn).stem}_input_boxes.png",
        )
        summary["question_answer"] = {
            "box_source": "FinTabNet Kaggle cell annotations transformed into rendered table-crop image coordinates.",
            "things_to_check": ["coordinate origin", "scale", "crop offset", "y-axis direction", "rendering resolution", "text order"],
        }
        summaries.append(summary)
    return summaries


def write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        "# Text Region Alignment Debug",
        "",
        "## Key Question",
        "",
        "GT PubTabNet cell boxes/text are the original PubTabNet annotations. Our annotation-derived TFLOP input boxes/text are the adapter output that converts the non-empty annotated cells into TFLOP's OCR-style input format (`bbox`, `text`, `score`). For PTN-Val, coordinates and text should match for non-empty cells; empty/no-text cells are not passed as TFLOP text-region inputs.",
        "",
        "## Outputs",
    ]
    for section, items in report["sections"].items():
        md.append(f"### {section}")
        for item in items:
            md.append(f"- `{item['output']}`")
    (OUT_DIR / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    report = {
        "output_dir": str(OUT_DIR),
        "sections": {
            "ptn_val": build_ptn_val_plots(),
            "ptn_test": build_ptn_test_plots(),
            "ftn": build_ftn_plots(),
        },
    }
    write_report(report)
    print(json.dumps({"output_dir": str(OUT_DIR), "readme": str(OUT_DIR / "README.md")}, indent=2))


if __name__ == "__main__":
    main()
