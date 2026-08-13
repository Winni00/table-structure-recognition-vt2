#!/bin/bash
set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling

/cluster/home/trinhwin/vt2/docling/.venv/bin/python -u \
  /cluster/home/trinhwin/vt2/docling/tableformer_fintabnet_kaggle_repro.py \
  --table-jsonl /cluster/home/trinhwin/vt2/docling/data/fintabnet_kaggle/FinTabNet_1.0.0_cell_val.jsonl \
  --limit 20 \
  --render-scale 2.0 \
  --device cpu \
  --num-threads 1 \
  --html-mode html_seq_cellmap \
  --output-dir /cluster/home/trinhwin/vt2/docling/results/tableformer_fintabnet_kaggle/smoke_20_html_seq_cellmap_local
