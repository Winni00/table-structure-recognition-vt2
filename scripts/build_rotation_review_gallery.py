"""Build an interactive orientation-review gallery for the 31 rotated cases."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
SUMMARY = ROOT / "results/tflop_paper_collection_updated_crops_rotation_broad_badcases_public/rotation_summary_gt_canon.json"
VARIANT_DIR = ROOT / "results/tflop_paper_collection_updated_crops_rotation_broad_badcases_public/images"
OUT = ROOT / "results/paper_collection_rotation_manual_review_31"


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> None:
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))["rows"]
    selected = [
        row
        for row in rows
        if int(row["best_rotation"]) != 0 and float(row["delta_best_vs_rot0_teds"]) > 0.01
    ]
    selected.sort(key=lambda row: float(row["delta_best_vs_rot0_teds"]), reverse=True)
    if len(selected) != 31:
        raise RuntimeError(f"Expected 31 review cases, found {len(selected)}")

    images_dir = OUT / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    cards = []

    for index, row in enumerate(selected, start=1):
        variants = {int(item["rotation"]): item for item in row["variants"]}
        image_links = {}
        for rotation in (0, 90, 270):
            source = VARIANT_DIR / variants[rotation]["filename"]
            target = images_dir / f"{index:02d}_rot{rotation}.png"
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(os.path.relpath(source, target.parent))
            image_links[rotation] = f"images/{target.name}"

        item = {
            "index": index,
            "filename": row["base_filename"],
            "teds_selected_rotation": row["best_rotation"],
            "teds_gain": row["delta_best_vs_rot0_teds"],
            "correct_rotation": "",
            "notes": "",
            "variants": row["variants"],
        }
        manifest.append(item)

        variant_html = []
        for rotation in (0, 90, 270):
            variant = variants[rotation]
            variant_html.append(
                f"""
                <label class="variant">
                  <input type="radio" name="rotation-{index}" value="{rotation}">
                  <span class="choice">Select {rotation}&deg;</span>
                  <span class="score">TEDS-S {variant['teds_s']:.3f} | TEDS {variant['teds']:.3f}</span>
                  <img src="{image_links[rotation]}" alt="{rotation} degree variant">
                </label>
                """
            )
        cards.append(
            f"""
            <section class="case" data-index="{index}" data-filename="{esc(row['base_filename'])}">
              <header>
                <strong>{index}/31</strong>
                <code>{esc(row['base_filename'])}</code>
                <span>Old TEDS choice: {row['best_rotation']}&deg; ({row['delta_best_vs_rot0_teds']:+.3f})</span>
              </header>
              <div class="variants">{''.join(variant_html)}</div>
              <label class="notes">Notes <input type="text" placeholder="optional"></label>
            </section>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rotation review: 31 Paper Collection tables</title>
<style>
  body {{ margin: 0; font-family: Arial, sans-serif; color: #171717; background: #f4f5f6; }}
  .toolbar {{ position: sticky; top: 0; z-index: 5; padding: 12px 20px; background: white; border-bottom: 1px solid #bbb; display: flex; gap: 16px; align-items: center; }}
  button {{ padding: 9px 14px; font-weight: 700; cursor: pointer; }}
  main {{ max-width: 1700px; margin: 18px auto; padding: 0 18px 60px; }}
  .intro {{ background: white; padding: 14px; border: 1px solid #ccc; margin-bottom: 18px; }}
  .case {{ background: white; border: 1px solid #aaa; margin: 0 0 22px; }}
  .case.unlabelled {{ border-left: 6px solid #c62828; }}
  .case header {{ display: flex; gap: 16px; align-items: center; padding: 10px 12px; border-bottom: 1px solid #ccc; flex-wrap: wrap; }}
  .case code {{ overflow-wrap: anywhere; }}
  .variants {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; padding: 8px; }}
  .variant {{ border: 3px solid transparent; padding: 6px; min-width: 0; cursor: pointer; }}
  .variant:has(input:checked) {{ border-color: #16813c; background: #effaf2; }}
  .variant input {{ width: 20px; height: 20px; vertical-align: middle; }}
  .choice {{ font-weight: 700; margin-left: 5px; }}
  .score {{ float: right; color: #666; font-size: 13px; }}
  .variant img {{ display: block; width: 100%; height: 430px; object-fit: contain; background: #fafafa; margin-top: 6px; }}
  .notes {{ display: block; padding: 10px 14px; border-top: 1px solid #ddd; }}
  .notes input {{ width: min(900px, 80%); padding: 6px; margin-left: 8px; }}
  @media (max-width: 900px) {{ .variants {{ grid-template-columns: 1fr; }} .variant img {{ height: 360px; }} }}
</style>
</head>
<body>
<div class="toolbar">
  <strong id="progress">0/31 labelled</strong>
  <button id="next-unlabelled">Jump to unlabelled</button>
  <button id="download">Download rotation_labels.csv</button>
  <span>Choose the orientation in which the table text is horizontally readable. Ignore TEDS when choosing.</span>
</div>
<main>
  <div class="intro">
    <h1>Manual orientation review</h1>
    <p>Review the 31 cases previously rotated because of a higher TEDS score. Select 0&deg;, 90&deg;, or 270&deg; based only on visual readability. Choices are saved in this browser.</p>
  </div>
  {''.join(cards)}
</main>
<script>
const key = 'paper-collection-rotation-review-v1';
const saved = JSON.parse(localStorage.getItem(key) || '{{}}');
const cases = [...document.querySelectorAll('.case')];
function update() {{
  const data = {{}};
  for (const c of cases) {{
    const selected = c.querySelector('input[type=radio]:checked');
    c.classList.toggle('unlabelled', !selected);
    data[c.dataset.filename] = {{correct_rotation: selected ? selected.value : '', notes: c.querySelector('.notes input').value}};
  }}
  localStorage.setItem(key, JSON.stringify(data));
  document.getElementById('progress').textContent = `${{Object.values(data).filter(x => x.correct_rotation !== '').length}}/31 labelled`;
}}
for (const c of cases) {{
  const row = saved[c.dataset.filename] || {{}};
  if (row.correct_rotation !== undefined && row.correct_rotation !== '') {{
    const input = c.querySelector(`input[value="${{row.correct_rotation}}"]`); if (input) input.checked = true;
  }}
  c.querySelector('.notes input').value = row.notes || '';
  c.addEventListener('change', update);
  c.querySelector('.notes input').addEventListener('input', update);
}}
update();
document.getElementById('next-unlabelled').addEventListener('click', () => {{
  const missing = cases.find(c => !c.querySelector('input[type=radio]:checked'));
  if (missing) missing.scrollIntoView({{behavior:'smooth', block:'start'}});
}});
document.getElementById('download').addEventListener('click', () => {{
  update();
  const data = JSON.parse(localStorage.getItem(key));
  const lines = ['filename,correct_rotation,notes'];
  for (const c of cases) {{
    const row = data[c.dataset.filename];
    const quote = value => `"${{String(value || '').replaceAll('"','""')}}"`;
    lines.push([quote(c.dataset.filename), quote(row.correct_rotation), quote(row.notes)].join(','));
  }}
  const blob = new Blob([lines.join('\\n')], {{type:'text/csv'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'rotation_labels.csv'; a.click(); URL.revokeObjectURL(a.href);
}});
</script>
</body>
</html>
"""

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / "review_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (OUT / "rotation_labels_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "correct_rotation", "notes"])
        writer.writeheader()
        for row in manifest:
            writer.writerow({"filename": row["filename"], "correct_rotation": "", "notes": ""})
    (OUT / "README.md").write_text(
        "# Manual rotation review\n\n"
        "Open `index.html`, select the visually correct orientation for all 31 tables, and click **Download rotation_labels.csv**.\n\n"
        "Selection criterion: table text must be horizontally readable. TEDS scores are context only and must not determine the label.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), "cases": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
