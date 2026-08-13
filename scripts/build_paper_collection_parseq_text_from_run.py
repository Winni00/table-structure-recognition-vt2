"""Replace MASTER-recognized text with PARSeq text for a TFLOP run directory.

This keeps the existing PSENet text-region boxes unchanged and only changes
the recognized text stored in ``aux_rec.pkl``. It is therefore a fair
recognizer-only comparison against MASTER:

    same table crops -> same PSENet boxes -> PARSeq text -> TFLOP inference
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RUN_DIR = ROOT / "results/tflop_paper_collection_updated_crops_broadbestrot_master_public"
DEFAULT_OUT_DIR = ROOT / "results/tflop_paper_collection_updated_crops_broadbestrot_parseq_public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fallback-to-master", action="store_true", default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def as_bbox(item: dict[str, Any]) -> list[float] | None:
    bbox = item.get("bbox")
    if bbox is None:
        return None
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    if len(bbox) == 4:
        return [float(v) for v in bbox]
    if len(bbox) >= 8:
        xs = [float(v) for v in bbox[0::2]]
        ys = [float(v) for v in bbox[1::2]]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def crop_region(image: Image.Image, bbox: list[float]) -> Image.Image | None:
    x0, y0, x1, y1 = bbox
    left = max(0, int(round(x0)))
    top = max(0, int(round(y0)))
    right = min(image.width, int(round(x1)))
    bottom = min(image.height, int(round(y1)))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def flush_batch(model, tokenizer, transform, device, batch: list[Image.Image]) -> tuple[list[str], list[float]]:
    if not batch:
        return [], []
    tensors = torch.stack([transform(img) for img in batch]).to(device)
    with torch.inference_mode():
        logits = model(tensors)
        probs = logits.softmax(-1)
        labels, confidences = tokenizer.decode(probs)
    texts = [normalize_text(label) for label in labels]
    scores: list[float] = []
    for conf in confidences:
        if hasattr(conf, "numel") and conf.numel() > 0:
            scores.append(float(conf.mean().detach().cpu()))
        else:
            scores.append(0.0)
    return texts, scores


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    for filename in ["aux.json", "subset.txt", "manifest.json"]:
        src = args.run_dir / filename
        if src.exists():
            shutil.copy2(src, out / filename)
    image_link = out / "images"
    if image_link.exists() or image_link.is_symlink():
        image_link.unlink() if image_link.is_symlink() or image_link.is_file() else shutil.rmtree(image_link)
    image_link.symlink_to((args.run_dir / "images").resolve())

    with (args.run_dir / "aux_rec.pkl").open("rb") as f:
        aux_rec: dict[str, list[dict[str, Any]]] = pickle.load(f)

    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
    model = model.eval().to(args.device)
    from strhub.data.module import SceneTextDataModule

    transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    aux_rec_parseq: dict[str, list[dict[str, Any]]] = {}
    reports: list[dict[str, Any]] = []
    batch_imgs: list[Image.Image] = []
    batch_refs: list[tuple[str, int, dict[str, Any], str]] = []
    parseq_text_by_ref: dict[tuple[str, int], tuple[str, float]] = {}

    def flush() -> None:
        nonlocal batch_imgs, batch_refs
        texts, scores = flush_batch(model, model.tokenizer, transform, args.device, batch_imgs)
        for (filename, idx, _item, _master_text), text, score in zip(batch_refs, texts, scores):
            parseq_text_by_ref[(filename, idx)] = (text, score)
        batch_imgs = []
        batch_refs = []

    total_regions = 0
    invalid_regions = 0
    filenames = sorted(aux_rec)
    if args.max_samples is not None:
        filenames = filenames[: args.max_samples]
        aux_rec = {filename: aux_rec[filename] for filename in filenames}

    for filename in filenames:
        image = Image.open(args.run_dir / "images" / filename).convert("RGB")
        for idx, item in enumerate(aux_rec[filename]):
            total_regions += 1
            bbox = as_bbox(item)
            master_text = normalize_text(str(item.get("text", "")))
            crop = crop_region(image, bbox) if bbox is not None else None
            if crop is None:
                invalid_regions += 1
                parseq_text_by_ref[(filename, idx)] = ("", 0.0)
                continue
            batch_imgs.append(crop)
            batch_refs.append((filename, idx, item, master_text))
            if len(batch_imgs) >= args.batch_size:
                flush()
        image.close()
    flush()

    changed = 0
    nonempty = 0
    fallback = 0
    for filename in filenames:
        out_items = []
        for idx, item in enumerate(aux_rec[filename]):
            master_text = normalize_text(str(item.get("text", "")))
            parseq_text, parseq_score = parseq_text_by_ref.get((filename, idx), ("", 0.0))
            used_text = parseq_text
            if not used_text and args.fallback_to_master:
                used_text = master_text
                fallback += 1
            if parseq_text:
                nonempty += 1
            if parseq_text and parseq_text != master_text:
                changed += 1
            new_item = dict(item)
            new_item["master_text"] = master_text
            new_item["parseq_text"] = parseq_text
            new_item["text"] = used_text
            new_item["parseq_score"] = parseq_score
            out_items.append(new_item)
        aux_rec_parseq[filename] = out_items

        reports.append(
            {
                "filename": filename,
                "regions": len(out_items),
                "parseq_nonempty_regions": sum(1 for item in out_items if item.get("parseq_text")),
                "parseq_differs_from_master_regions": sum(
                    1
                    for item in out_items
                    if item.get("parseq_text") and item.get("parseq_text") != item.get("master_text")
                ),
            }
        )

    with (out / "aux_rec.pkl").open("wb") as f:
        pickle.dump(aux_rec_parseq, f)

    summary = {
        "source_run_dir": str(args.run_dir),
        "output_dir": str(out),
        "recognizer": "PARSeq",
        "device": args.device,
        "batch_size": args.batch_size,
        "samples": len(aux_rec_parseq),
        "regions": total_regions,
        "invalid_regions": invalid_regions,
        "parseq_nonempty_regions": nonempty,
        "parseq_nonempty_ratio": round(nonempty / max(1, total_regions), 4),
        "parseq_differs_from_master_regions": changed,
        "parseq_differs_ratio": round(changed / max(1, total_regions), 4),
        "fallback_to_master_regions": fallback,
    }
    (out / "parseq_text_report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    (out / "parseq_text_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
