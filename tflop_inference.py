"""TFLOP standalone inference runner for selected hard tables.

This script adapts the official TFLOP test-time inference path to the local
12-table benchmark setup used in this workspace.

Compared with the TableFormer runners, TFLOP needs one extra input source:
- OCR / recognition boxes with text for each table crop

Expected workflow:
1. Load the selected hard tables metadata JSON.
2. Load OCR entries for each `table_id`.
3. Load the TFLOP checkpoint and tokenizer.
4. Run the official model inference path on each table crop.
5. Save:
   - raw_output
   - normalized_output
   - ground_truth references
   - evaluation_result placeholder
   - debug visualizations

The OCR input is expected as one JSON file per table inside `--ocr-dir`:

    <ocr-dir>/<table_id>.ocr.json

Each file should contain a list like:

[
  {"bbox": [x1, y1, x2, y2], "text": "cell text"},
  ...
]

The checkpoint can be provided either as a local directory or as a Hugging Face
repo id. A local checkpoint directory is expected to contain at least:
- config.json
- pytorch_model.bin
- tokenizer files
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
TFLOP_REPO = (
    BASE_DIR / "repo" / "TFLOP_clean"
    if (BASE_DIR / "repo" / "TFLOP_clean").exists()
    else BASE_DIR / "repo" / "TFLOP"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "tflop_hard_tables"
DEFAULT_HARD_TABLES_JSON = (
    BASE_DIR / "data" / "paper_collection_meta" / "selected_hard_tables.json"
)
DEFAULT_TFLOP_CHECKPOINT_DIR = BASE_DIR / "models" / "tflop"
DEFAULT_OCR_DIR = BASE_DIR / "data" / "paper_collection_meta" / "tflop_ocr"

# Make the cloned repo importable so the adapter uses local TFLOP code.
if str(TFLOP_REPO) not in sys.path:
    sys.path.insert(0, str(TFLOP_REPO))

from tflop.datamodule.preprocess.common_utils import (  # noqa: E402
    int_convert_and_pad_coords,
    rescale_bbox,
    serialize_bbox_top_left_bottom_right,
)
from tflop.datamodule.preprocess.image_utils import prepare_image_tensor  # noqa: E402
from tflop.model.model.TFLOP import TFLOP  # noqa: E402
from tflop.model.model.TFLOP_Config import TFLOPConfig  # noqa: E402
from tflop.utils import custom_format_html, decode_OTSL_seq, resolve_missing_config  # noqa: E402


@dataclass
class OCRCell:
    """Represents one OCR text box expected by the TFLOP adapter."""

    bbox: list[float]
    text: str


class HardTablesTFLOPDataset(Dataset):
    """Minimal dataset wrapper that mirrors TFLOP's official test preprocessing."""

    def __init__(
        self,
        *,
        rows: list[dict],
        ocr_dir: Path,
        tokenizer,
        config,
    ) -> None:
        self.rows = rows
        self.ocr_dir = ocr_dir
        self.tokenizer = tokenizer
        self.config = config
        self.prompt_end_token = self.config.get("prompt_end_token", "<s_answer>")
        self.prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(
            self.prompt_end_token
        )
        self.input_img_size = (
            self.config.input_size.width,
            self.config.input_size.height,
        )
        self.bbox_token_cnt = self.config.get("bbox_token_cnt", None)
        self.max_length = self.config.get("max_length", None)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        table_id = row["table_id"]
        image_path = Path(row["input_png"])
        gt_html = Path(row["gt_html"]).read_text(encoding="utf-8")

        # OCR data is external to the table crop and must be prepared separately.
        ocr_path = self.ocr_dir / f"{table_id}.ocr.json"
        if not ocr_path.exists():
            raise FileNotFoundError(
                f"Missing OCR file for {table_id}: {ocr_path}. "
                "Create one JSON file per table crop before running TFLOP inference."
            )

        ocr_entries = json.loads(ocr_path.read_text(encoding="utf-8"))
        cells = [OCRCell(bbox=entry["bbox"], text=entry.get("text", "")) for entry in ocr_entries]

        with Image.open(image_path) as img:
            table_image = img.convert("RGB")
            image_tensor, org_img_size, padding_dims = prepare_image_tensor(
                input_image=table_image,
                target_img_size=self.input_img_size,
                random_padding=False,
            )

        bbox_coords = [cell.bbox for cell in cells]
        bbox_texts = [cell.text for cell in cells]
        rescaled_coords = rescale_bbox(
            list_of_coords=bbox_coords,
            org_img_size=org_img_size,
            new_img_size=self.input_img_size,
            padding_dims=padding_dims,
        )
        rescaled_coords, cell_texts, _ = serialize_bbox_top_left_bottom_right(
            rescaled_coords,
            bbox_texts,
        )

        if len(rescaled_coords) > (self.bbox_token_cnt - 1):
            rescaled_coords = rescaled_coords[: (self.bbox_token_cnt - 1)]
            cell_texts = cell_texts[: (self.bbox_token_cnt - 1)]

        padding_coord = [
            self.input_img_size[0] + 3,
            self.input_img_size[1] + 3,
            self.input_img_size[0] + 3,
            self.input_img_size[1] + 3,
        ]
        coords_int_padded = int_convert_and_pad_coords(
            coords=rescaled_coords,
            padding_coord=padding_coord,
            max_length=self.bbox_token_cnt,
        )
        valid_coord_length = torch.tensor(len(rescaled_coords))

        # TFLOP expects only the BOS + answer prompt at test time.
        input_parse = self.tokenizer.bos_token + self.prompt_end_token
        input_ids = self.tokenizer(
            input_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        input_ids = input_ids[: self.max_length - self.bbox_token_cnt]
        prompt_end_index = torch.nonzero(input_ids == self.prompt_end_token_id).sum()
        cell_text_collated = "<special_cell_text_sep>".join(cell_texts)

        return {
            "image_tensor": image_tensor,
            "input_ids": input_ids,
            "coords_int_padded": coords_int_padded,
            "valid_coord_length": valid_coord_length,
            "prompt_end_index": prompt_end_index,
            "html_with_content": gt_html,
            "cell_text_collated": cell_text_collated,
            "file_name": table_id,
            "image_path": str(image_path),
            "ocr_path": str(ocr_path),
            "metadata": row,
            "ocr_entries": ocr_entries,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone TFLOP inference on the selected hard table crops "
            "using OCR text boxes prepared outside the TFLOP repo."
        )
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_HARD_TABLES_JSON,
        help="Path to the selected hard tables metadata JSON.",
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=DEFAULT_OCR_DIR,
        help="Directory containing one <table_id>.ocr.json file per hard table.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_TFLOP_CHECKPOINT_DIR,
        help="Local TFLOP checkpoint directory containing config.json, pytorch_model.bin and tokenizer files.",
    )
    parser.add_argument(
        "--checkpoint-repo-id",
        default=None,
        help="Optional Hugging Face repo id to download the TFLOP checkpoint if no local checkpoint is present.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use, for example cuda or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size. Start with 1 for stability.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where TFLOP outputs will be written.",
    )
    return parser.parse_args()


def load_rows(dataset_json: Path) -> list[dict]:
    """Load the selected hard tables metadata rows."""
    return json.loads(dataset_json.read_text(encoding="utf-8"))


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    """Resolve a local TFLOP checkpoint path."""
    if args.checkpoint_path.exists():
        return args.checkpoint_path

    if args.checkpoint_repo_id is None:
        raise FileNotFoundError(
            f"Local TFLOP checkpoint not found: {args.checkpoint_path}. "
            "Provide --checkpoint-path or --checkpoint-repo-id."
        )

    from huggingface_hub import snapshot_download

    download_path = snapshot_download(repo_id=args.checkpoint_repo_id)
    return Path(download_path)


def load_exp_config() -> object:
    """Load the repo-default TFLOP experiment settings used for test-time preprocessing."""
    general_cfg = OmegaConf.load(TFLOP_REPO / "config" / "exp_configs" / "general_exp.yaml")
    data_cfg = OmegaConf.load(TFLOP_REPO / "config" / "exp_configs" / "data_pubtabnet.yaml")
    return OmegaConf.merge(general_cfg, data_cfg)


def custom_load_state_dict(model, state_dict_map):
    """Mirror the checkpoint loading logic used by TFLOP's official test.py."""
    assert len(state_dict_map) == 2
    if state_dict_map["key"] == "encoder":
        model.encoder.load_state_dict(state_dict_map["value"])
    elif state_dict_map["key"] == "decoder":
        model.decoder.load_state_dict(state_dict_map["value"])
    else:
        raise ValueError("Invalid state dict map key")


def load_model_and_tokenizer(checkpoint_path: Path, exp_config, device: str):
    """Load the TFLOP tokenizer and model using the official test.py logic."""
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

    model_config_raw = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
    model_config_dict = {
        k: model_config_raw.get(k, exp_config.get(k))
        for k in TFLOPConfig.get_member_variables()
        if model_config_raw.get(k, exp_config.get(k)) is not None
    }
    model_config_dict = resolve_missing_config(model_config_dict)

    model = TFLOP(
        config=TFLOPConfig(**model_config_dict),
        tokenizer=tokenizer,
        data_ids=["C-tag"],
    )

    saved_state_dict = torch.load(
        checkpoint_path / "pytorch_model.bin",
        map_location=torch.device("cpu"),
    )
    encoder_state_dict = {
        k[len("encoder.") :]: v
        for k, v in saved_state_dict.items()
        if k.startswith("encoder.")
    }
    decoder_state_dict = {
        k[len("decoder.") :]: v
        for k, v in saved_state_dict.items()
        if k.startswith("decoder.")
    }
    if len(saved_state_dict) != (len(encoder_state_dict) + len(decoder_state_dict)):
        raise ValueError("Invalid saved state dict")

    with ThreadPool(2) as pool:
        pool.map(
            partial(custom_load_state_dict, model),
            [
                {"key": "encoder", "value": encoder_state_dict},
                {"key": "decoder", "value": decoder_state_dict},
            ],
        )

    torch_dtype = model_config_raw.get("torch_dtype", "float16")
    if torch_dtype == "float16":
        model.half()
    elif torch_dtype == "bfloat16":
        model.bfloat16()

    model = model.to(device)
    model.eval()
    return model, tokenizer, torch_dtype


def make_json_safe(value):
    """Convert tensors and numpy values to plain Python."""
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def render_ocr_visualization(image_path: Path, ocr_entries: list[dict], output_path: Path) -> None:
    """Draw OCR text boxes used by TFLOP for debugging and reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for entry in ocr_entries:
            x1, y1, x2, y2 = entry["bbox"]
            draw.rectangle((x1, y1, x2, y2), outline="orange", width=2)
        canvas.save(output_path)


def normalize_tflop_output(pred_string: str) -> dict:
    """Project TFLOP output into the shared comparison structure."""
    full_html = (
        pred_string
        if pred_string.startswith("<html>")
        else f"<html><body><table>{pred_string}</table></body></html>"
    )
    return {
        "cells": [],
        "structure": {
            "html": full_html,
        },
    }


def collate_batch(batch: list[dict]) -> dict:
    """Simple collate function aligned with TFLOP's test-time batch structure."""
    return {
        "image_tensor": torch.stack([item["image_tensor"] for item in batch], dim=0),
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "coords_int_padded": torch.stack([item["coords_int_padded"] for item in batch], dim=0),
        "valid_coord_length": torch.stack([item["valid_coord_length"] for item in batch], dim=0),
        "prompt_end_index": torch.stack([item["prompt_end_index"] for item in batch], dim=0),
        "html_with_content": [item["html_with_content"] for item in batch],
        "cell_text_collated": [item["cell_text_collated"] for item in batch],
        "file_name": [item["file_name"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "ocr_path": [item["ocr_path"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
        "ocr_entries": [item["ocr_entries"] for item in batch],
    }


def run_inference(model, tokenizer, dataloader, torch_dtype: str, output_dir: Path) -> None:
    """Run TFLOP inference and persist one prediction JSON per table."""
    summary = []
    for batch in dataloader:
        image_tensors = batch["image_tensor"]
        decoder_input_ids = batch["input_ids"]
        coord_input_idx = batch["coords_int_padded"]
        coord_input_length = batch["valid_coord_length"]
        prompt_end_idxs = batch["prompt_end_index"]
        html_with_content = batch["html_with_content"]
        cell_texts = batch["cell_text_collated"]
        file_names = batch["file_name"]

        pointer_args = {
            "coord_input_idx": coord_input_idx,
            "coord_input_length": coord_input_length,
        }

        decoder_prompts = torch.nn.utils.rnn.pad_sequence(
            [
                input_id[: end_idx + 1]
                for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)
            ],
            batch_first=True,
        )

        if torch_dtype == "float16":
            image_tensors = image_tensors.half()
        elif torch_dtype == "bfloat16":
            image_tensors = image_tensors.bfloat16()

        image_tensors = image_tensors.to(model.device)
        decoder_prompts = decoder_prompts.to(model.device)
        pointer_args["coord_input_idx"] = pointer_args["coord_input_idx"].to(model.device)
        pointer_args["coord_input_length"] = pointer_args["coord_input_length"].to(model.device)

        with torch.no_grad():
            preds = model.inference(
                image_tensors=image_tensors,
                prompt_tensors=decoder_prompts,
                return_json=False,
                return_attentions=False,
                pointer_args=pointer_args,
            )

        for data_i, file_name in enumerate(file_names):
            token_id_seq = preds["output_sequences"][data_i]
            token_seq = tokenizer.convert_ids_to_tokens(token_id_seq)
            cell_text_data = cell_texts[data_i].split("<special_cell_text_sep>")
            decoded_html_seq = decode_OTSL_seq(
                otsl_token_seq=token_seq,
                pointer_tensor=preds["text_to_dr_coord"][data_i],
                cell_text_data=cell_text_data,
            )
            pred_string, full_pred_html = custom_format_html(decoded_html_seq, tokenizer)

            answer_string = html_with_content[data_i]
            normalized_output = normalize_tflop_output(pred_string)

            viz_path = output_dir / "viz" / f"{file_name}.ocr.png"
            render_ocr_visualization(
                Path(batch["image_path"][data_i]),
                batch["ocr_entries"][data_i],
                viz_path,
            )

            sample_output = {
                "sample": file_name,
                "model_version": "tflop",
                "checkpoint_ref": str(model.config.name_or_path),
                "image_path": batch["image_path"][data_i],
                "ocr_path": batch["ocr_path"][data_i],
                "raw_output": {
                    "pred_string": pred_string,
                    "answer_string": answer_string,
                    "full_pred_html": full_pred_html,
                    "token_pred": token_seq,
                },
                "normalized_output": normalized_output,
                "debug_visualizations": [str(viz_path)],
                "evaluation_result": None,
                "dataset_metadata": batch["metadata"][data_i],
                "ground_truth": {
                    "gt_html": batch["metadata"][data_i].get("gt_html"),
                    "gt_xml": batch["metadata"][data_i].get("gt_xml"),
                },
            }

            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{file_name}.pred.json"
            output_path.write_text(
                json.dumps(make_json_safe(sample_output), indent=2),
                encoding="utf-8",
            )

            summary.append(
                {
                    "sample": file_name,
                    "output_file": str(output_path),
                    "debug_visualizations": [str(viz_path)],
                }
            )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset_json)
    exp_config = load_exp_config()
    checkpoint_path = resolve_checkpoint_path(args)
    model, tokenizer, torch_dtype = load_model_and_tokenizer(
        checkpoint_path,
        exp_config,
        args.device,
    )

    dataset = HardTablesTFLOPDataset(
        rows=rows,
        ocr_dir=args.ocr_dir,
        tokenizer=tokenizer,
        config=exp_config,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    run_inference(model, tokenizer, dataloader, torch_dtype, args.output_dir)
    print("TFLOP inference finished.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"OCR dir: {args.ocr_dir}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
