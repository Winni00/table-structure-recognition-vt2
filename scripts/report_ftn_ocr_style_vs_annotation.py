#!/usr/bin/env python3
"""Report and visualize FTN OCR-style vs annotation-derived TFLOP inputs."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
OCR_RUN = ROOT / "results" / "tflop_fintabnet_ocr_style_full_10622"
ANN_RUN = ROOT / "results" / "tflop_fintabnet_full_annotation_excl_problem_pages"
OUT_DIR = ROOT / "results" / "ftn_ocr_style_vs_annotation_report"


def load_teds(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        row[0]: {
            "filename": row[0],
            "pred": row[1],
            "gt": row[2],
            "edit_distance": float(row[3]),
            "teds_s": float(row[4]),
            "teds": float(row[5]),
        }
        for row in rows
    }


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rec(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("rb") as f:
        return pickle.load(f)


def clean_text(text: str, limit: int = 34) -> str:
    text = re.sub(r"<[^>]+>", "", str(text or ""))
    text = " ".join(text.split())
    text = text.encode("ascii", "replace").decode("ascii")
    return text[:limit] + ("..." if len(text) > limit else "")


def draw_boxes_panel(
    image: Image.Image,
    items: list[dict[str, Any]],
    title: str,
    score_line: str,
    color: tuple[int, int, int],
    max_labels: int = 80,
) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    scale = min(1.8, max(1.0, 760 / max(w, 1)))
    im = image.resize((int(w * scale), int(h * scale)))
    top = 92
    canvas = Image.new("RGB", (im.width, im.height + top + 8), "white")
    canvas.paste(im, (0, top))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    draw.text((8, 28), score_line, fill=(70, 70, 70), font=font)
    draw.text((8, 48), "Origin: top-left; bbox=[left, top, right, bottom]", fill=(90, 90, 90), font=font)
    draw.text((8, 68), f"Text regions: {len(items)}", fill=color, font=font)
    draw.rectangle([0, top, im.width - 1, top + im.height - 1], outline=(35, 95, 210), width=3)
    draw.ellipse([0, top, 9, top + 9], fill=(220, 35, 35))
    draw.line([5, top + 5, min(im.width - 8, 95), top + 5], fill=(220, 35, 35), width=2)
    draw.line([5, top + 5, 5, min(top + im.height - 8, top + 70)], fill=(220, 35, 35), width=2)
    labels = 0
    for item in items:
        bbox = item.get("bbox")
        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()
        if bbox is None or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [float(v) * scale for v in bbox]
        y0 += top
        y1 += top
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        label = clean_text(item.get("text", ""))
        if label and labels < max_labels and (x1 - x0 > 18):
            ly = max(top, y0 - 11)
            lx = max(0, min(x0, im.width - 160))
            draw.rectangle([lx, ly, min(im.width, lx + len(label) * 6 + 8), ly + 11], fill=(255, 255, 255))
            draw.text((lx + 2, ly), label, fill=color, font=font)
            labels += 1
    return canvas


def render_comparison(
    filename: str,
    group: str,
    ann: dict[str, Any],
    ocr: dict[str, Any],
    ann_rec: list[dict[str, Any]],
    ocr_rec: list[dict[str, Any]],
) -> dict[str, Any]:
    img_path = ANN_RUN / "images" / filename
    base = Image.open(img_path).convert("RGB")
    ann_panel = draw_boxes_panel(
        base,
        ann_rec,
        "Previous FTN setup: annotation-derived cell boxes/text",
        f"official TFLOP eval: TEDS-S={ann['teds_s']:.4f}, TEDS={ann['teds']:.4f}",
        (30, 150, 70),
    )
    ocr_panel = draw_boxes_panel(
        base,
        ocr_rec,
        "New FTN setup: OCR-style PSENet+MASTER text regions",
        f"official TFLOP eval: TEDS-S={ocr['teds_s']:.4f}, TEDS={ocr['teds']:.4f}",
        (180, 55, 170),
    )
    gap = 24
    width = ann_panel.width + ocr_panel.width + gap
    height = max(ann_panel.height, ocr_panel.height)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(ann_panel, (0, 0))
    canvas.paste(ocr_panel, (ann_panel.width + gap, 0))
    out_dir = OUT_DIR / "examples" / group
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(filename).stem}_ann_vs_ocr.png"
    canvas.save(out_path)
    html_path = out_dir / f"{Path(filename).stem}_html_scores.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}pre{white-space:pre-wrap;border:1px solid #ccc;padding:12px;max-height:440px;overflow:auto}</style>"
        f"<h1>{filename}</h1>"
        f"<p>Group: {group}</p>"
        f"<p>Annotation official: TEDS-S={ann['teds_s']:.6f}, TEDS={ann['teds']:.6f}</p>"
        f"<p>OCR-style official: TEDS-S={ocr['teds_s']:.6f}, TEDS={ocr['teds']:.6f}</p>"
        f"<p>Delta OCR-Annotation: TEDS-S={ocr['teds_s']-ann['teds_s']:.6f}, TEDS={ocr['teds']-ann['teds']:.6f}</p>"
        "<h2>Annotation prediction HTML</h2><pre>"
        + ann["pred"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre><h2>OCR-style prediction HTML</h2><pre>"
        + ocr["pred"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre><h2>GT HTML</h2><pre>"
        + ocr["gt"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre>",
        encoding="utf-8",
    )
    return {
        "filename": filename,
        "group": group,
        "png": str(out_path),
        "html": str(html_path),
        "annotation_regions": len(ann_rec),
        "ocr_regions": len(ocr_rec),
        "annotation_teds_s": ann["teds_s"],
        "annotation_teds": ann["teds"],
        "ocr_teds_s": ocr["teds_s"],
        "ocr_teds": ocr["teds"],
        "delta_teds_s": ocr["teds_s"] - ann["teds_s"],
        "delta_teds": ocr["teds"] - ann["teds"],
    }


def select_examples(ann_scores: dict[str, dict[str, Any]], ocr_scores: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    common = sorted(set(ann_scores) & set(ocr_scores))
    records = []
    for fn in common:
        ann = ann_scores[fn]
        ocr = ocr_scores[fn]
        records.append(
            {
                "filename": fn,
                "delta_teds": ocr["teds"] - ann["teds"],
                "delta_teds_s": ocr["teds_s"] - ann["teds_s"],
                "ann_teds": ann["teds"],
                "ocr_teds": ocr["teds"],
            }
        )
    improvements = sorted(records, key=lambda r: (-r["delta_teds"], r["filename"]))[:5]
    drops = sorted(records, key=lambda r: (r["delta_teds"], r["filename"]))[:5]
    similar = sorted(records, key=lambda r: (abs(r["delta_teds"]), -max(r["ann_teds"], r["ocr_teds"]), r["filename"]))[:5]
    good_ocr = sorted([r for r in records if r["ocr_teds"] >= 0.9], key=lambda r: (-r["ocr_teds"], r["filename"]))[:5]
    return {
        "largest_ocr_improvements": [r["filename"] for r in improvements],
        "largest_ocr_drops": [r["filename"] for r in drops],
        "similar_scores": [r["filename"] for r in similar],
        "high_ocr_score": [r["filename"] for r in good_ocr],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ann_scores = load_teds(ANN_RUN / "ted_score_output.json")
    ocr_scores = load_teds(OCR_RUN / "ted_score_output.json")
    ftn_can_scores = load_teds(ANN_RUN / "ted_score_output_ftn_canonicalized.json")
    ann_rec = load_rec(ANN_RUN / "aux_rec.pkl")
    ocr_rec = load_rec(OCR_RUN / "aux_rec.pkl")
    ocr_summary = load_summary(OCR_RUN / "ted_score_output.summary.json")
    report_json = load_summary(OCR_RUN / "analysis_report" / "report.json")

    groups = select_examples(ann_scores, ocr_scores)
    examples = []
    for group, filenames in groups.items():
        for fn in filenames:
            examples.append(render_comparison(fn, group, ann_scores[fn], ocr_scores[fn], ann_rec.get(fn, []), ocr_rec.get(fn, [])))

    runs = report_json["runs"]
    summary = {
        "output_dir": str(OUT_DIR),
        "scores": runs,
        "ocr_summary": ocr_summary,
        "ocr_validation": report_json.get("ocr_validation"),
        "comparisons": {
            "ocr_minus_annotation_official_common_samples": report_json["comparisons"]["annotation_official_to_ocr_official"],
            "ocr_minus_annotation_ftn_canonicalized_common_samples": report_json["comparisons"]["annotation_canonicalized_to_ocr_official"],
        },
        "examples": examples,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# FTN OCR-style vs Annotation-input Report",
        "",
        "## Score comparison",
        "",
        "| Run | Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "ftn_ocr_style_official_tflop_eval": "FTN OCR-style, official TFLOP eval",
        "ftn_annotation_input_official_tflop_eval": "FTN annotation input, official TFLOP eval",
        "ftn_annotation_input_ftn_canonicalized_diagnostic": "FTN annotation input, FTN-canonicalized diagnostic",
    }
    for key in [
        "ftn_ocr_style_official_tflop_eval",
        "ftn_annotation_input_official_tflop_eval",
        "ftn_annotation_input_ftn_canonicalized_diagnostic",
    ]:
        row = runs[key]
        md.append(
            f"| {labels[key]} | {row['samples']:,} | {row['teds_s']*100:.2f} | {row['teds']*100:.2f} | {row['teds_s_1']:,} | {row['teds_1']:,} |"
        )
    comp = report_json["comparisons"]["annotation_official_to_ocr_official"]
    comp_can = report_json["comparisons"]["annotation_canonicalized_to_ocr_official"]
    md += [
        "",
        "## Findings",
        "",
        f"- OCR-style improves over the previous annotation-input official TFLOP run by **{comp['mean_delta_teds_s']*100:.2f} TEDS-S** and **{comp['mean_delta_teds']*100:.2f} TEDS** on {comp['common_samples']:,} common samples.",
        f"- However, OCR-style is below the FTN-canonicalized diagnostic by **{abs(comp_can['mean_delta_teds_s'])*100:.2f} TEDS-S** and **{abs(comp_can['mean_delta_teds'])*100:.2f} TEDS** on the same common samples.",
        f"- OCR-style is better/worse/same than annotation-input official by TEDS on **{comp['right_better_teds']:,} / {comp['right_worse_teds']:,} / {comp['right_same_teds']:,}** samples.",
        "- Two samples were excluded from OCR-style inference/eval because PSENet+MASTER produced zero text regions: `fintabnet_01635_105722.png`, `fintabnet_01722_105588.png`.",
        "- OCR validation found no invalid or out-of-bounds boxes for the remaining OCR regions.",
        "",
        "## Interpretation",
        "",
        "The OCR-style input test partially supports the input-distribution hypothesis: TFLOP performs better when FTN is represented with OCR-like text regions than with raw annotation cell boxes under the strict official TFLOP evaluator. But the improvement is small and does not close the paper gap. The stronger FTN-canonicalized diagnostic suggests that evaluation/HTML convention and checkpoint/training mismatch remain central issues.",
        "",
        "## Visual examples",
        "",
        "Each PNG shows the same table crop side by side: left = previous annotation-derived cell boxes/text, right = OCR-style PSENet+MASTER text regions.",
    ]
    for group in groups:
        md.append(f"### {group}")
        for item in [e for e in examples if e["group"] == group]:
            md.append(
                f"- `{item['filename']}`: ΔTEDS={item['delta_teds']*100:.2f}, "
                f"annotation={item['annotation_teds']*100:.2f}, OCR={item['ocr_teds']*100:.2f}; "
                f"[PNG]({item['png']}) [HTML]({item['html']})"
            )
        md.append("")
    (OUT_DIR / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR), "readme": str(OUT_DIR / "README.md"), "examples": len(examples)}, indent=2))


if __name__ == "__main__":
    main()
