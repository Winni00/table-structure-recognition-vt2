"""Build a readable HTML gallery for existing PyMuPDF mapping visuals."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
RESULT_DIR = ROOT / "results/pymupdf_extraction_verification"


def card(row: dict) -> str:
    visual = Path(row["visual"])
    relative_visual = Path("visuals") / visual.name
    return f"""
    <article>
      <h3>{html.escape(row['filename'])}</h3>
      <p>
        Page {row['page_number']} · rotation {row['rotation_used']}° ·
        {row['regions']} PSE regions · {row['pymupdf_empty_regions']} empty ·
        PyMuPDF/MASTER similarity {row['mean_similarity_pymupdf_vs_master']:.3f}
      </p>
      <a href="{html.escape(str(relative_visual))}">
        <img src="{html.escape(str(relative_visual))}" loading="lazy" alt="Mapping verification">
      </a>
    </article>
    """


def main() -> None:
    summary = json.loads((RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
    rows = summary["sample_summaries"]
    worst = [row for row in rows if row["group"] == "worst_pymupdf"]
    random = [row for row in rows if row["group"] == "random"]
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PyMuPDF mapping verification</title>
<style>
body {{ font: 16px Arial, sans-serif; max-width: 1500px; margin: 28px auto; padding: 0 20px; color: #222; }}
nav {{ position: sticky; top: 0; background: white; padding: 12px 0; border-bottom: 1px solid #bbb; }}
article {{ border-top: 1px solid #bbb; padding: 18px 0 28px; }}
img {{ display: block; max-width: 100%; max-height: 920px; border: 1px solid #ddd; }}
p {{ line-height: 1.45; }}
</style></head><body>
<h1>PyMuPDF mapping verification</h1>
<p>Left: table crop with original PSE boxes. Right: original PDF page with mapped PyMuPDF rectangles.</p>
<nav><a href="#worst">30 worst PyMuPDF cases</a> · <a href="#random">20 random cases</a></nav>
<h2 id="worst">30 worst PyMuPDF cases</h2>
{''.join(card(row) for row in worst)}
<h2 id="random">20 random cases</h2>
{''.join(card(row) for row in random)}
</body></html>"""
    target = RESULT_DIR / "gallery.html"
    target.write_text(page, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
