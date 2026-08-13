from pathlib import Path
import json
import re

raw_root = Path("/cluster/home/trinhwin/vt2/docling/data/paper_collection_raw")
meta_root = Path("/cluster/home/trinhwin/vt2/docling/data/paper_collection_meta")
meta_root.mkdir(parents=True, exist_ok=True)

out_json = meta_root / "table_inventory.json"
out_csv = meta_root / "table_inventory.csv"

rows = []

for paper_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
    paper_id = paper_dir.name
    paper_pdf = paper_dir / f"{paper_id}.pdf"
    paper_xml = paper_dir / f"{paper_id}.xml"

    png_files = sorted(paper_dir.glob(f"{paper_id}_table_*.png"))

    for png in png_files:
        m = re.match(rf"^{re.escape(paper_id)}_table_(\d+)\.png$", png.name)
        if not m:
            continue

        table_idx = int(m.group(1))
        html = paper_dir / f"{paper_id}_table_{table_idx}.html"
        xml = paper_dir / f"{paper_id}_table_{table_idx}.xml"

        rows.append({
            "paper_id": paper_id,
            "table_id": f"{paper_id}_table_{table_idx}",
            "table_index": table_idx,
            "input_png": str(png),
            "gt_html": str(html) if html.exists() else None,
            "gt_xml": str(xml) if xml.exists() else None,
            "paper_pdf": str(paper_pdf) if paper_pdf.exists() else None,
            "paper_xml": str(paper_xml) if paper_xml.exists() else None,
        })

out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

with out_csv.open("w", encoding="utf-8") as f:
    f.write("paper_id,table_id,table_index,input_png,gt_html,gt_xml,paper_pdf,paper_xml\n")
    for r in rows:
        vals = [
            r["paper_id"],
            r["table_id"],
            str(r["table_index"]),
            r["input_png"] or "",
            r["gt_html"] or "",
            r["gt_xml"] or "",
            r["paper_pdf"] or "",
            r["paper_xml"] or "",
        ]
        f.write(",".join(f'"{v}"' for v in vals) + "\n")

print(f"Wrote {len(rows)} tables")
print(out_json)
print(out_csv)