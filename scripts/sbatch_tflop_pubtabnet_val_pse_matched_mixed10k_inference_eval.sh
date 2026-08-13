#!/bin/bash
#SBATCH --job-name=ptnval_mix10k_eval
#SBATCH --time=8:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
PY_SYS=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
INPUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_val_pse_matched_filtered_8958
SAVE_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_val_pse_matched_filtered_8958_mixed10k_lr1e5
EXP_CONFIG=/cluster/home/trinhwin/vt2/docling/results/tflop_mixed_ftn16k_ptn4k_train_10k_lr1e5/tflop_mixed_ftn16k_ptn4k/from_public_checkpoint_10000steps_lr1e5/config.yaml

mkdir -p "$SAVE_DIR"
ln -sfn "$INPUT_DIR/aux.json" "$SAVE_DIR/aux.json"
ln -sfn "$INPUT_DIR/aux_rec.pkl" "$SAVE_DIR/aux_rec.pkl"
cp "$INPUT_DIR/summary.json" "$SAVE_DIR/input_summary.json"

MODEL_DIR=$($PY_SYS scripts/find_latest_mixed_ftn_ptn_checkpoint.py)
MODEL_CONFIG="${MODEL_DIR}/config_infer_float16.json"
$PY_SYS - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
data["torch_dtype"] = "float16"
(model_dir / "config_infer_float16.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
PY

$PY $TFLOP_REPO/test.py \
  --tokenizer_name_or_path "$MODEL_DIR" \
  --model_name_or_path "$MODEL_DIR" \
  --exp_config_path "$EXP_CONFIG" \
  --model_config_path "$MODEL_CONFIG" \
  --aux_json_path "$INPUT_DIR/aux.json" \
  --aux_img_path /cluster/home/trinhwin/vt2/docling/data/TFLOP-dataset/images/validation \
  --aux_rec_pkl_path "$INPUT_DIR/aux_rec.pkl" \
  --batch_size 1 \
  --save_dir "$SAVE_DIR" \
  --use_validation

$PY /cluster/home/trinhwin/vt2/docling/repo/TFLOP_clean/evaluate_ted.py \
  --model_inference_pathdir "$SAVE_DIR" \
  --output_savepath "$SAVE_DIR" \
  --legacy-strip-cell-contents

$PY_SYS - <<'PY'
import json
from pathlib import Path

run = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_val_pse_matched_filtered_8958_mixed10k_lr1e5")
rows = json.loads((run / "ted_score_output.json").read_text(encoding="utf-8"))
summary = {
    "samples": len(rows),
    "teds_s": sum(r[-2] for r in rows) / len(rows),
    "teds": sum(r[-1] for r in rows) / len(rows),
    "teds_s_1": sum(r[-2] == 1.0 for r in rows),
    "teds_1": sum(r[-1] == 1.0 for r in rows),
}
(run / "ted_score_output.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(run / "README.md").write_text(
    "# PTN-Val PSE-matched Eval: Mixed FTN+PTN 10K LR=1e-5\n\n"
    "| Samples | TEDS-S | TEDS | TEDS-S=1 | TEDS=1 |\n"
    "|---:|---:|---:|---:|---:|\n"
    f"| {summary['samples']} | {summary['teds_s']:.4f} | {summary['teds']:.4f} | {summary['teds_s_1']} | {summary['teds_1']} |\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
PY
