from pathlib import Path
import json

inventory_path = Path("/cluster/home/trinhwin/vt2/docling/data/paper_collection_meta/table_inventory.json")
difficulty_path = Path("/cluster/home/trinhwin/vt2/docling/data/paper_collection_meta/manual_table_difficulty.json")
out_path = Path("/cluster/home/trinhwin/vt2/docling/data/paper_collection_meta/selected_hard_tables.json")

inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
difficulty = json.loads(difficulty_path.read_text(encoding="utf-8"))

inv_by_table_id = {row["table_id"]: row for row in inventory}

selected = []
missing = []

for row in difficulty:
    if row["difficulty"] != "hard":
        continue

    table_id = row["table_id"]
    inv = inv_by_table_id.get(table_id)

    if inv is None:
        missing.append(table_id)
        continue

    merged = {
        "paper_id": inv["paper_id"],
        "table_id": inv["table_id"],
        "table_index": inv["table_index"],
        "difficulty": row["difficulty"],
        "input_png": inv["input_png"],
        "gt_html": inv["gt_html"],
        "gt_xml": inv["gt_xml"],
        "paper_pdf": inv["paper_pdf"],
        "paper_xml": inv["paper_xml"],
    }
    selected.append(merged)

out_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")

print(f"Wrote {len(selected)} selected hard tables to:")
print(out_path)

if missing:
    print("\nMissing table_ids:")
    for m in missing:
        print(m)
else:
    print("\nNo missing table_ids.")
