#!/usr/bin/env python3
"""Replace MASTER-recognized text with TrOCR text for a TFLOP run directory.

This keeps the existing PSENet text-region boxes unchanged and only changes
the recognized text stored in ``aux_rec.pkl``:

    same table crops -> same PSENet boxes -> TrOCR text -> TFLOP inference
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
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
DEFAULT_RUN_DIR = ROOT / "results/tflop_paper_collection_updated_crops_broadbestrot_master_public"
DEFAULT_OUT_DIR = ROOT / "results/tflop_paper_collection_updated_crops_broadbestrot_trocr_public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-name", default="microsoft/trocr-base-printed")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fallback-to-master", action="store_true", default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
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


def flush_batch(
    *,
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    device: str,
    batch: list[Image.Image],
    max_new_tokens: int,
) -> list[str]:
    if not batch:
        return []
    inputs = processor(images=batch, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    with torch.inference_mode():
        generated = model.generate(pixel_values, max_new_tokens=max_new_tokens)
    return [normalize_text(text) for text in processor.batch_decode(generated, skip_special_tokens=True)]


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

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    model = model.eval().to(args.device)

    filenames = sorted(aux_rec)
    if args.max_samples is not None:
        filenames = filenames[: args.max_samples]
        aux_rec = {filename: aux_rec[filename] for filename in filenames}

    batch_imgs: list[Image.Image] = []
    batch_refs: list[tuple[str, int]] = []
    trocr_text_by_ref: dict[tuple[str, int], str] = {}

    def flush() -> None:
        nonlocal batch_imgs, batch_refs
        texts = flush_batch(
            model=model,
            processor=processor,
            device=args.device,
            batch=batch_imgs,
            max_new_tokens=args.max_new_tokens,
        )
        for ref, text in zip(batch_refs, texts):
            trocr_text_by_ref[ref] = text
        batch_imgs = []
        batch_refs = []

    total_regions = 0
    invalid_regions = 0
    for filename in filenames:
        image = Image.open(args.run_dir / "images" / filename).convert("RGB")
        for idx, item in enumerate(aux_rec[filename]):
            total_regions += 1
            bbox = as_bbox(item)
            crop = crop_region(image, bbox) if bbox is not None else None
            if crop is None:
                invalid_regions += 1
                trocr_text_by_ref[(filename, idx)] = ""
                continue
            batch_imgs.append(crop)
            batch_refs.append((filename, idx))
            if len(batch_imgs) >= args.batch_size:
                flush()
        image.close()
    flush()

    aux_rec_trocr: dict[str, list[dict[str, Any]]] = {}
    reports: list[dict[str, Any]] = []
    changed = 0
    nonempty = 0
    fallback = 0
    for filename in filenames:
        out_items = []
        for idx, item in enumerate(aux_rec[filename]):
            master_text = normalize_text(str(item.get("text", "")))
            trocr_text = normalize_text(trocr_text_by_ref.get((filename, idx), ""))
            used_text = trocr_text
            if not used_text and args.fallback_to_master:
                used_text = master_text
                fallback += 1
            if trocr_text:
                nonempty += 1
            if trocr_text and trocr_text != master_text:
                changed += 1
            new_item = dict(item)
            new_item["master_text"] = master_text
            new_item["trocr_text"] = trocr_text
            new_item["text"] = used_text
            new_item["trocr_score"] = 0.0
            out_items.append(new_item)
        aux_rec_trocr[filename] = out_items
        reports.append(
            {
                "filename": filename,
                "regions": len(out_items),
                "trocr_nonempty_regions": sum(1 for item in out_items if item.get("trocr_text")),
                "trocr_differs_from_master_regions": sum(
                    1
                    for item in out_items
                    if item.get("trocr_text") and item.get("trocr_text") != item.get("master_text")
                ),
            }
        )

    with (out / "aux_rec.pkl").open("wb") as f:
        pickle.dump(aux_rec_trocr, f)

    summary = {
        "source_run_dir": str(args.run_dir),
        "output_dir": str(out),
        "recognizer": "TrOCR",
        "model_name": args.model_name,
        "device": args.device,
        "batch_size": args.batch_size,
        "samples": len(aux_rec_trocr),
        "regions": total_regions,
        "invalid_regions": invalid_regions,
        "trocr_nonempty_regions": nonempty,
        "trocr_nonempty_ratio": round(nonempty / max(1, total_regions), 4),
        "trocr_differs_from_master_regions": changed,
        "trocr_differs_ratio": round(changed / max(1, total_regions), 4),
        "fallback_to_master_regions": fallback,
    }
    (out / "trocr_text_report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    (out / "trocr_text_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
