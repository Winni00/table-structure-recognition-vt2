#!/usr/bin/env python3
"""Create a visual report for PubTabNet samples with TEDS close to, but below, 1."""

from __future__ import annotations

import difflib
import html as html_escape
import json
import re
import shutil
from pathlib import Path
from typing import Any

from lxml import html
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
RUN_DIR = ROOT / "results" / "tableformer_pubtabnet_hf_original" / "val_full"
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


def normalized_table(prediction: dict[str, Any]) -> dict[str, Any]:
    normalized = prediction.get("normalized_output", {})
    if isinstance(normalized, list):
        return normalized[0] if normalized else {}
    return normalized if isinstance(normalized, dict) else {}


def tokenise_html(text: str) -> list[str]:
    return [tok for tok in re.split(r"(<[^>]+>)", text) if tok and tok.strip()]


def table_stats(text: str) -> dict[str, int]:
    root = html.fromstring(ensure_html_table_document(text))
    table = root.xpath("body/table")
    if not table:
        return {"nodes": 0, "rows": 0, "cells": 0}
    t = table[0]
    return {
        "nodes": len(t.xpath(".//*")),
        "rows": len(t.xpath(".//tr")),
        "cells": len(t.xpath(".//td|.//th")),
    }


def first_token_diffs(pred_html: str, gt_html: str, limit: int = 14) -> list[str]:
    pred_tokens = tokenise_html(pred_html)
    gt_tokens = tokenise_html(gt_html)
    sm = difflib.SequenceMatcher(a=pred_tokens, b=gt_tokens)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        pred_part = " ".join(pred_tokens[i1:i2])
        gt_part = " ".join(gt_tokens[j1:j2])
        out.append(f"{tag}: pred[{i1}:{i2}]={pred_part!r} gt[{j1}:{j2}]={gt_part!r}")
        if len(out) >= limit:
            break
    return out


def write_html_page(path: Path, title: str, body: str) -> None:
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html_escape.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1d1d1f; }}
table {{ border-collapse: collapse; margin: 8px 0 24px; }}
td, th {{ border: 1px solid #555; padding: 3px 6px; vertical-align: top; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.sample {{ border-top: 2px solid #222; padding-top: 18px; margin-top: 22px; }}
iframe {{ width: 100%; height: 340px; border: 1px solid #999; background: white; }}
img {{ max-width: 100%; border: 1px solid #aaa; }}
pre {{ white-space: pre-wrap; background: #f6f6f6; padding: 12px; overflow: auto; }}
.muted {{ color: #666; }}
</style>
</head>
<body>
{body}
</body>
</html>
""",
        encoding="utf-8",
    )


def render_overview(samples: list[dict[str, Any]], out_path: Path) -> None:
    font = ImageFont.load_default()
    card_w, card_h = 520, 330
    cols = 2
    rows = (len(samples) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * card_w, rows * card_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, sample in enumerate(samples):
        x = (idx % cols) * card_w
        y = (idx // cols) * card_h
        draw.rectangle([x + 8, y + 8, x + card_w - 8, y + card_h - 8], outline=(30, 30, 30), width=2)
        draw.text((x + 18, y + 18), sample["sample"], fill=(0, 0, 0), font=font)
        draw.text(
            (x + 18, y + 36),
            f"TEDS-S={sample['teds_s']:.6f}  TEDS={sample['teds']:.6f}",
            fill=(0, 0, 0),
            font=font,
        )
        draw.text((x + 18, y + 54), sample["short_issue"], fill=(120, 0, 0), font=font)
        image_path = Path(sample["image_path"])
        if image_path.exists():
            im = Image.open(image_path).convert("RGB")
            im.thumbnail((card_w - 36, card_h - 92), Image.Resampling.LANCZOS)
            sheet.paste(im, (x + 18, y + 82))
    sheet.save(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads((RUN_DIR / "per_table.json").read_text(encoding="utf-8"))
    candidates = [r for r in rows if float(r["teds_s"]) < 1.0]
    candidates.sort(key=lambda r: (float(r["teds_s"]), float(r["teds"])), reverse=True)
    selected = candidates[:10]

    report_sections = [
        "<h1>Near-1 TEDS PubTabNet examples after canonicalization</h1>",
        f"<p class='muted'>Run: {RUN_DIR}</p>",
        "<p>Selected by highest TEDS-S below 1.0 after PubTabNet-style section canonicalization.</p>",
    ]
    overview_samples: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for rank, row in enumerate(selected, start=1):
        sample = row["sample"]
        pred_path = Path(row["pred_path"])
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        norm = normalized_table(pred)
        pred_html = canonicalize_pubtabnet_sections(norm.get("structure", {}).get("html", ""))
        gt_html = canonicalize_pubtabnet_sections(pred.get("ground_truth", {}).get("gt_html", ""))
        stats_pred = table_stats(pred_html)
        stats_gt = table_stats(gt_html)
        diffs = first_token_diffs(pred_html, gt_html)

        sample_dir = OUT_DIR / f"{rank:02d}_{sample}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "prediction_canonicalized.html").write_text(pred_html, encoding="utf-8")
        (sample_dir / "ground_truth.html").write_text(gt_html, encoding="utf-8")
        (sample_dir / "token_diff.txt").write_text("\n".join(diffs) + "\n", encoding="utf-8")
        metrics = {
            "rank": rank,
            "sample": sample,
            "filename": row.get("filename"),
            "teds": row["teds"],
            "teds_s": row["teds_s"],
            "pred_stats": stats_pred,
            "gt_stats": stats_gt,
            "pred_path": str(pred_path),
            "image_path": pred.get("image_path"),
            "first_diffs": diffs,
        }
        (sample_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        image_path = Path(pred.get("image_path", ""))
        image_name = ""
        if image_path.exists():
            image_name = image_path.name
            shutil.copy2(image_path, sample_dir / image_name)

        short_issue = diffs[0][:120] if diffs else "No token diff found; likely parser/tree nuance"
        overview_samples.append(
            {
                "sample": sample,
                "teds": float(row["teds"]),
                "teds_s": float(row["teds_s"]),
                "image_path": str(image_path),
                "short_issue": short_issue,
            }
        )
        manifest.append(metrics)

        image_html = f"<img src='{rank:02d}_{sample}/{html_escape.escape(image_name)}'>" if image_name else ""
        diff_html = html_escape.escape("\n".join(diffs))
        report_sections.append(
            f"""
<section class="sample">
<h2>{rank}. {html_escape.escape(sample)}</h2>
<p><b>TEDS-S:</b> {float(row['teds_s']):.6f} &nbsp; <b>TEDS:</b> {float(row['teds']):.6f}</p>
<p><b>Pred stats:</b> {stats_pred} &nbsp; <b>GT stats:</b> {stats_gt}</p>
{image_html}
<div class="grid">
  <div><h3>Prediction canonicalized</h3><iframe src="{rank:02d}_{sample}/prediction_canonicalized.html"></iframe></div>
  <div><h3>Ground truth</h3><iframe src="{rank:02d}_{sample}/ground_truth.html"></iframe></div>
</div>
<h3>First token diffs</h3>
<pre>{diff_html}</pre>
</section>
"""
        )

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_html_page(OUT_DIR / "index.html", "Near-1 TEDS PubTabNet examples", "\n".join(report_sections))
    render_overview(overview_samples, OUT_DIR / "overview_near_one_top10.png")
    print(f"Wrote {OUT_DIR}")
    for item in manifest:
        print(
            f"{item['rank']:02d} {item['sample']} "
            f"TEDS-S={item['teds_s']:.6f} TEDS={item['teds']:.6f}"
        )


if __name__ == "__main__":
    main()
