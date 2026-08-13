#!/usr/bin/env python3
"""Render near-1 PubTabNet TEDS examples in the pre/post image style."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from render_tableformer_pre_post_examples import (
    FONT_SMALL,
    FONT_TITLE,
    bbox_from_dict,
    bbox_from_list,
    crop_from_boxes,
    draw_boxes_panel,
    render_html_preview,
)


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
RUN_DIR = ROOT / "results" / "tableformer_pubtabnet_hf_original" / "val_full"
HF_PUBTABNET_JSONL = ROOT / "data" / "pubtabnet_hf" / "extracted" / "pubtabnet" / "PubTabNet_2.0.0.jsonl"
OUT_DIR = (
    ROOT
    / "results"
    / "tableformer_examples"
    / "near_one_teds_pubtabnet_canonicalized_top10"
)


def ensure_html_table_document(text: str) -> str:
    cleaned = text.strip().replace("<?xml version='1.0' encoding='UTF-8'?>", "").strip()
    lower = cleaned.lower()
    if "<html" in lower:
        return cleaned
    if "<table" in lower:
        return f"<html><body>{cleaned}</body></html>"
    return f"<html><body><table>{cleaned}</table></body></html>"


def canonicalize_pubtabnet_sections(text: str) -> str:
    html_doc = ensure_html_table_document(text)
    return re.sub(
        r"(</thead>)((?:<tr>.*?</tr>)+)(</table>)",
        r"\1<tbody>\2</tbody>\3",
        html_doc,
        flags=re.DOTALL,
    )


def rich_text(tokens: list[str]) -> str:
    return "".join(tokens or [])


def html_from_pubtabnet_row(row: dict[str, Any]) -> str:
    html = row.get("html")
    if isinstance(html, str):
        return ensure_html_table_document(html)
    if not isinstance(html, dict):
        return ""
    tokens = html.get("structure", {}).get("tokens", [])
    cells = iter(html.get("cells", []))
    parts: list[str] = []
    for token in tokens:
        if token == "</td>":
            cell = next(cells, {})
            parts.append(rich_text(cell.get("tokens", [])))
            parts.append(token)
        else:
            parts.append(token)
    return ensure_html_table_document("".join(parts))


def load_hf_gt(filenames: set[str]) -> dict[str, str]:
    gt: dict[str, str] = {}
    with HF_PUBTABNET_JSONL.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            filename = row.get("filename")
            if filename in filenames:
                gt[filename] = html_from_pubtabnet_row(row)
                if len(gt) == len(filenames):
                    break
    missing = sorted(filenames - set(gt))
    if missing:
        raise RuntimeError(f"Missing HF PubTabNet GT rows: {missing}")
    return gt


def render_one(pred_path: Path, row: dict, out_dir: Path, hf_gt_html: str) -> None:
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    image = Image.open(payload["image_path"]).convert("RGB")
    raw = payload["raw_output"][0]
    details = raw["predict_details"]

    pre_boxes = [bbox_from_list(b) for b in details.get("prediction_bboxes_page", [])]
    pre_boxes = [b for b in pre_boxes if b]
    post_boxes = [bbox_from_dict(cell.get("bbox", {})) for cell in raw.get("tf_responses", [])]
    post_boxes = [b for b in post_boxes if b]
    crop = crop_from_boxes(pre_boxes + post_boxes, image.size)

    pre = draw_boxes_panel(
        image,
        crop,
        "Before matching/postprocessing: raw model cell boxes",
        pre_boxes,
        "#7b2cbf",
    )
    post = draw_boxes_panel(
        image,
        crop,
        "After matching/postprocessing: tf_responses with matched content",
        post_boxes,
        "#198754",
    )
    gt_html = canonicalize_pubtabnet_sections(hf_gt_html)
    pred_html = canonicalize_pubtabnet_sections(
        payload["normalized_output"][0]["structure"].get("html", "")
    )
    gt_html_panel = render_html_preview(
        "Original HF PubTabNet GT HTML",
        gt_html,
        max_rows=999,
        max_cols=16,
    )
    pred_html_panel = render_html_preview(
        "Canonicalized prediction HTML used for TEDS",
        pred_html,
        max_rows=999,
        max_cols=16,
    )

    width = max(pre.width + post.width + 24, gt_html_panel.width, pred_html_panel.width)
    height = 94 + max(pre.height, post.height) + gt_html_panel.height + pred_html_panel.height + 28
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{payload['sample']} | canonicalized TEDS-S={float(row['teds_s']):.6f}, "
        f"TEDS={float(row['teds']):.6f}"
    )
    draw.text((8, 8), title, font=FONT_TITLE, fill="black")
    draw.text(
        (8, 42),
        "Purple = raw model boxes. Green = matched content boxes. GT preview is rebuilt from original HF PubTabNet jsonl.",
        font=FONT_SMALL,
        fill="gray",
    )
    canvas.paste(pre, (0, 94))
    canvas.paste(post, (pre.width + 24, 94))
    html_y = 94 + max(pre.height, post.height) + 12
    canvas.paste(gt_html_panel, (0, html_y))
    canvas.paste(pred_html_panel, (0, html_y + gt_html_panel.height + 8))
    canvas.save(out_dir / f"{payload['sample']}_pre_post.png")


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = json.loads((RUN_DIR / "per_table.json").read_text(encoding="utf-8"))
    candidates = []
    for row in rows:
        teds_s = float(row["teds_s"])
        if not 0.94 <= teds_s <= 0.965:
            continue
        payload = json.loads(Path(row["pred_path"]).read_text(encoding="utf-8"))
        norm = payload["normalized_output"][0]
        num_rows = int(norm.get("num_rows") or 999)
        num_cols = int(norm.get("num_cols") or 999)
        num_cells = len(norm.get("cells", []))
        if num_rows <= 12 and num_cols <= 8 and num_cells <= 80:
            candidates.append(
                (
                    abs(teds_s - 0.955),
                    num_rows * num_cols,
                    num_cells,
                    row,
                )
            )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = [item[3] for item in candidates[:10]]
    hf_gt_by_filename = load_hf_gt({row["filename"] for row in selected})

    manifest = []
    for rank, row in enumerate(selected, start=1):
        pred_path = Path(row["pred_path"])
        render_one(pred_path, row, OUT_DIR, hf_gt_by_filename[row["filename"]])
        manifest.append(
            {
                "rank": rank,
                "sample": row["sample"],
                "filename": row["filename"],
                "teds_s": row["teds_s"],
                "teds": row["teds"],
                "pred_path": str(pred_path),
            }
        )
        print(f"{rank:02d} {row['sample']} TEDS-S={row['teds_s']:.6f} TEDS={row['teds']:.6f}")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
