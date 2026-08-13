#!/cluster/home/trinhwin/vt2/docling/.venv/bin/python
"""Run TableFormer PubTabNet with the local-v3 empty-cell bbox reconstruction.

This wrapper intentionally keeps ``tableformer_pubtabnet_repro.py`` and
``tableformer_pubtabnet_repro_bbox_experiment.py`` unchanged. It monkey-patches
only the reconstruction function so we can compare local-v3 against the stable
baseline without drifting the blueprint script.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path("/cluster/home/trinhwin/vt2/docling")
BASE_SCRIPT = ROOT / "tableformer_pubtabnet_repro_bbox_experiment.py"
V3_SCRIPT = ROOT / "scripts/experiment_pubtabnet_bbox_reconstruction_local_v3.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = load_module(BASE_SCRIPT, "tableformer_pubtabnet_repro_bbox_experiment_v3_run")
    local_v3 = load_module(V3_SCRIPT, "pubtabnet_bbox_local_v3")
    base.reconstruct_missing_cell_bboxes = local_v3.reconstruct_missing_cell_bboxes_local_v3
    base.main()


if __name__ == "__main__":
    main()
