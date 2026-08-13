#!/usr/bin/env python3
"""Print latest checkpoint directory for the mixed FTN+PTN 10K run."""

from pathlib import Path
import re

ROOT = Path(
    "/cluster/home/trinhwin/vt2/docling/results/"
    "tflop_mixed_ftn16k_ptn4k_train_10k_lr1e5/"
    "tflop_mixed_ftn16k_ptn4k/from_public_checkpoint_10000steps_lr1e5"
)

candidates = []
for path in ROOT.glob("epoch_*_step_*"):
    if not (path / "pytorch_model.bin").exists() or not (path / "config.json").exists():
        continue
    match = re.search(r"_step_(\d+)$", path.name)
    step = int(match.group(1)) if match else -1
    candidates.append((step, path))
if not candidates:
    raise SystemExit(f"No checkpoint found under {ROOT}")
print(max(candidates, key=lambda item: item[0])[1])
