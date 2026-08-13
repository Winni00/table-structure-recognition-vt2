#!/usr/bin/env python3
"""Single-GPU TFLOP training entrypoint for small sanity runs."""

from __future__ import annotations

import argparse
import datetime
import importlib
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.plugins import CheckpointIO


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
TFLOP_ROOT = BASE_DIR / "repo" / "TFLOP"
sys.path.insert(0, str(TFLOP_ROOT))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tflop.lightning_module.lightning_module import DataPLModule, TFLOPModelPLModule  # noqa: E402
from tflop.utils import (  # noqa: E402
    save_config_file,
    set_seed,
    set_up_logger_and_callbacks,
    set_up_tokenizer,
)


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        del checkpoint["state_dict"]
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, storage_options=None):
        checkpoint = torch.load(path + "artifacts.ckpt")
        state_dict = torch.load(path + "pytorch_model.bin")
        checkpoint["state_dict"] = {"model." + key: value for key, value in state_dict.items()}
        return checkpoint

    def remove_checkpoint(self, path) -> None:
        return super().remove_checkpoint(path)


def train(config):
    set_seed(config.get("seed", 42))
    assert any(
        [
            config.get("use_RowWise_contLearning", False),
            config.get("use_ColWise_contLearning", False),
        ]
    ), "Contrastive Learning setting is not correct."

    tokenizer = set_up_tokenizer(
        pretrained_tokenizer_name_or_path=config.get(
            "pretrained_tokenizer_name_or_path", "hyunwoongko/asian-bart-ecjk"
        ),
        bbox_special_tokens=config.bbox_special_tokens,
        other_special_tokens=config.special_chars,
    )

    model_module = TFLOPModelPLModule(config=config, tokenizer=tokenizer, mode="train")
    data_module = DataPLModule(config=config)

    dataset_class = getattr(
        importlib.import_module(config.dataset_script_path), config.dataset_class_name
    )
    data_module.train_dataset = dataset_class(tokenizer=tokenizer, split="train", config=config)
    data_module.val_dataset = dataset_class(tokenizer=tokenizer, split="validation", config=config)

    logger, lr_callback, checkpoint_callback, bar = set_up_logger_and_callbacks(config)
    strategy = config.get("strategy", "auto")
    if strategy == "none":
        strategy = "auto"

    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        devices=1,
        strategy=strategy,
        accelerator="gpu",
        plugins=CustomCheckpointIO(),
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        precision=config.get("precision", "bf16"),
        num_sanity_val_steps=config.get("num_sanity_val_steps", 1),
        logger=logger,
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        callbacks=[lr_callback, checkpoint_callback, bar],
    )
    trainer.fit(
        model_module,
        data_module,
        ckpt_path=config.get("resume_from_checkpoint_path", None),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, required=True)
    parser.add_argument("--data_config", type=str, required=True)
    args, left_argv = parser.parse_known_args()

    exp_config = OmegaConf.load(args.exp_config)
    data_config = OmegaConf.load(args.data_config)
    cli_config = OmegaConf.from_cli(left_argv)
    config = OmegaConf.unsafe_merge(exp_config, data_config, cli_config)
    config.exp_version = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not config.exp_version
        else config.exp_version
    )
    config.bbox_special_tokens = [
        f"<bbox_{i}>" for i in range(max(config.input_size.values()) + 1)
    ]

    OmegaConf.resolve(config)
    for sanity_config in ["result_path", "exp_name", "exp_version"]:
        assert config.get(sanity_config, None) is not None, f"{sanity_config} is not set"

    save_config_file(config, Path(config.result_path) / config.exp_name / config.exp_version)
    print(OmegaConf.to_yaml(config))
    train(config)


if __name__ == "__main__":
    main()
