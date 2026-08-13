#!/bin/bash
#SBATCH --job-name=tflop_ftn_overfit_eval
#SBATCH --time=02:00:00
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=cai_exp
#SBATCH --partition=gpu
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%j_%x.err

set -euo pipefail

cd /cluster/home/trinhwin/vt2/docling
mkdir -p logs

PY_TFLOP=/cluster/home/trinhwin/vt2/docling/.venv-tflop39/bin/python
PY_BUILD=/cluster/home/trinhwin/vt2/docling/.venv/bin/python
TFLOP_REPO=/cluster/home/trinhwin/vt2/docling/repo/TFLOP
SUBSET=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_overfit100_eval_subset
EXP_CONFIG=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_overfit100_training/tflop_fintabnet_overfit100/from_public_checkpoint_300steps/config.yaml
MODEL_CONFIG=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_overfit100_training/tflop_fintabnet_overfit100/from_public_checkpoint_300steps/epoch_5_step_300/config_infer_float16.json

BASE_OUT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_overfit100_eval_baseline_public
FT_OUT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_overfit100_eval_epoch5_step300
mkdir -p "$BASE_OUT" "$FT_OUT"

$PY_TFLOP $TFLOP_REPO/test.py \
  --tokenizer_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --model_name_or_path /cluster/home/trinhwin/vt2/docling/models/tflop \
  --exp_config_path "$EXP_CONFIG" \
  --model_config_path "$MODEL_CONFIG" \
  --aux_json_path "$SUBSET/aux.json" \
  --aux_img_path "$SUBSET/images" \
  --aux_rec_pkl_path "$SUBSET/aux_rec.pkl" \
  --batch_size 1 \
  --save_dir "$BASE_OUT" \
  --use_validation

$PY_TFLOP $TFLOP_REPO/evaluate_ted.py \
  --model_inference_pathdir "$BASE_OUT" \
  --output_savepath "$BASE_OUT"

$PY_BUILD scripts/rescore_tflop_fintabnet_canonicalized.py \
  --run-dir "$BASE_OUT" \
  --output-name ted_score_output_ftn_canonicalized.json \
  --num-processes 8

$PY_TFLOP $TFLOP_REPO/test.py \
  --tokenizer_name_or_path /cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_overfit100_training/tflop_fintabnet_overfit100/from_public_checkpoint_300steps/epoch_5_step_300 \
  --model_name_or_path /cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_overfit100_training/tflop_fintabnet_overfit100/from_public_checkpoint_300steps/epoch_5_step_300 \
  --exp_config_path "$EXP_CONFIG" \
  --model_config_path "$MODEL_CONFIG" \
  --aux_json_path "$SUBSET/aux.json" \
  --aux_img_path "$SUBSET/images" \
  --aux_rec_pkl_path "$SUBSET/aux_rec.pkl" \
  --batch_size 1 \
  --save_dir "$FT_OUT" \
  --use_validation

$PY_TFLOP $TFLOP_REPO/evaluate_ted.py \
  --model_inference_pathdir "$FT_OUT" \
  --output_savepath "$FT_OUT"

$PY_BUILD scripts/rescore_tflop_fintabnet_canonicalized.py \
  --run-dir "$FT_OUT" \
  --output-name ted_score_output_ftn_canonicalized.json \
  --num-processes 8

$PY_BUILD - <<'PY'
import json
from pathlib import Path

base = Path("/cluster/home/trinhwin/vt2/docling/results")
runs = {
    "baseline_public": base / "tflop_fintabnet_overfit100_eval_baseline_public",
    "finetuned_epoch5_step300": base / "tflop_fintabnet_overfit100_eval_epoch5_step300",
}

def summarize_official(path: Path):
    rows = json.loads((path / "ted_score_output.json").read_text(encoding="utf-8"))
    return {
        "samples": len(rows),
        "teds_s": sum(r[-2] for r in rows) / len(rows),
        "teds": sum(r[-1] for r in rows) / len(rows),
        "teds_s_1": sum(r[-2] == 1.0 for r in rows),
        "teds_1": sum(r[-1] == 1.0 for r in rows),
    }

summary = {}
for name, path in runs.items():
    summary[name] = {
        "official": summarize_official(path),
        "ftn_canonicalized": json.loads(
            (path / "ted_score_output_ftn_canonicalized.summary.json").read_text(
                encoding="utf-8"
            )
        ),
    }
out = base / "tflop_fintabnet_overfit100_eval_summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
