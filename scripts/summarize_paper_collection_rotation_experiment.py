"""Summarize the Paper Collection rotation-candidate TFLOP experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RUN_DIR = ROOT / "results/tflop_paper_collection_updated_crops_rotation_candidates_public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--scores-json", type=Path, default=None)
    parser.add_argument("--output-name", default="rotation_summary.json")
    parser.add_argument("--readme-name", default="README.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores_path = args.scores_json or (args.run_dir / "ted_score_output.json")
    manifest_path = args.run_dir / "rotation_manifest.json"
    if not scores_path.exists():
        raise SystemExit(f"Missing scores: {scores_path}")
    scores = {
        row[0]: {"teds_s": float(row[-2]), "teds": float(row[-1])}
        for row in json.loads(scores_path.read_text(encoding="utf-8"))
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    by_base: dict[str, list[dict[str, Any]]] = {}
    for item in manifest:
        filename = item["filename"]
        if filename not in scores:
            continue
        row = {**item, **scores[filename]}
        by_base.setdefault(item["base_filename"], []).append(row)

    rows = []
    for base, variants in sorted(by_base.items()):
        variants.sort(key=lambda row: row["rotation"])
        baseline = next((row for row in variants if row["rotation"] == 0), None)
        best = max(variants, key=lambda row: (row["teds"], row["teds_s"]))
        rows.append(
            {
                "base_filename": base,
                "best_rotation": best["rotation"],
                "best_teds_s": best["teds_s"],
                "best_teds": best["teds"],
                "rot0_teds_s": baseline["teds_s"] if baseline else None,
                "rot0_teds": baseline["teds"] if baseline else None,
                "delta_best_vs_rot0_teds_s": (
                    best["teds_s"] - baseline["teds_s"] if baseline else None
                ),
                "delta_best_vs_rot0_teds": (
                    best["teds"] - baseline["teds"] if baseline else None
                ),
                "aspect_ratio_h_over_w": best["aspect_ratio_h_over_w"],
                "vertical_box_ratio": best["vertical_box_ratio"],
                "variants": [
                    {
                        "rotation": row["rotation"],
                        "filename": row["filename"],
                        "teds_s": row["teds_s"],
                        "teds": row["teds"],
                    }
                    for row in variants
                ],
            }
        )

    improved = [
        row
        for row in rows
        if row["best_rotation"] != 0
        and row["delta_best_vs_rot0_teds"] is not None
        and row["delta_best_vs_rot0_teds"] > 0.01
    ]
    worsened_or_same = [row for row in rows if row not in improved]
    summary = {
        "run_dir": str(args.run_dir),
        "base_candidates": len(rows),
        "variant_samples": len(scores),
        "rotation_helped_count": len(improved),
        "rotation_not_helpful_count": len(worsened_or_same),
        "mean_best_teds_s": sum(row["best_teds_s"] for row in rows) / len(rows) if rows else 0.0,
        "mean_best_teds": sum(row["best_teds"] for row in rows) / len(rows) if rows else 0.0,
        "mean_rot0_teds_s": sum(row["rot0_teds_s"] for row in rows if row["rot0_teds_s"] is not None)
        / max(1, sum(row["rot0_teds_s"] is not None for row in rows)),
        "mean_rot0_teds": sum(row["rot0_teds"] for row in rows if row["rot0_teds"] is not None)
        / max(1, sum(row["rot0_teds"] is not None for row in rows)),
    }
    (args.run_dir / args.output_name).write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Rotation Candidate Experiment",
        "",
        "Goal: test whether rotating likely vertical/portrait table crops before PSENet+MASTER improves TFLOP TEDS.",
        "",
        "Each selected table was evaluated as 0°, 90°, and 270° with the same GT HTML and public TFLOP checkpoint.",
        "",
        "## Summary",
        "",
        f"- Base candidates: `{summary['base_candidates']}`",
        f"- Variant samples: `{summary['variant_samples']}`",
        f"- Rotation helped (> +0.01 TEDS and best rotation != 0): `{summary['rotation_helped_count']}`",
        f"- Mean rot0 TEDS-S/TEDS: `{summary['mean_rot0_teds_s']:.4f}` / `{summary['mean_rot0_teds']:.4f}`",
        f"- Mean best TEDS-S/TEDS: `{summary['mean_best_teds_s']:.4f}` / `{summary['mean_best_teds']:.4f}`",
        "",
        "## Top Rotation Improvements",
        "",
        "| Base file | Best rot | rot0 TEDS | best TEDS | Delta TEDS | Aspect | Vertical boxes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        rows,
        key=lambda row: row["delta_best_vs_rot0_teds"] or 0.0,
        reverse=True,
    )[:20]:
        lines.append(
            f"| `{row['base_filename']}` | {row['best_rotation']} | "
            f"{(row['rot0_teds'] or 0.0):.4f} | {row['best_teds']:.4f} | "
            f"{(row['delta_best_vs_rot0_teds'] or 0.0):+.4f} | "
            f"{row['aspect_ratio_h_over_w']:.2f} | {row['vertical_box_ratio']:.2f} |"
        )
    (args.run_dir / args.readme_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
