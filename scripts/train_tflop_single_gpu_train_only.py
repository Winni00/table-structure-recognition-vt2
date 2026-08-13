#!/usr/bin/env python3
"""Single-GPU TFLOP training entrypoint without generation-time validation.

The official validation path calls autoregressive generation and can OOM on
32GB GPUs for long FTN sequences. For staged fine-tuning we only need training
checkpoints here; full validation is run afterwards through sharded inference
and TEDS.
"""

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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO


BASE_DIR = Path("/cluster/home/trinhwin/vt2/docling")
TFLOP_ROOT = BASE_DIR / "repo" / "TFLOP"
sys.path.insert(0, str(TFLOP_ROOT))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tflop.lightning_module.lightning_module import DataPLModule, TFLOPModelPLModule  # noqa: E402
from tflop.utils import save_config_file, set_seed, set_up_tokenizer  # noqa: E402


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

    result_path = Path(config.result_path)
    logger = TensorBoardLogger(
        save_dir=result_path,
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )
    ckpt_dir = result_path / config.exp_name / config.exp_version
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="artifacts-{epoch:02d}-{step:07d}",
        every_n_train_steps=config.get("save_every_n_train_steps", 5000),
        save_top_k=-1,
        save_last=True,
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

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
        gradient_clip_val=config.gradient_clip_val,
        precision=config.get("precision", "bf16"),
        num_sanity_val_steps=0,
        limit_val_batches=0,
        logger=logger,
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        callbacks=[lr_callback, checkpoint_callback],
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
