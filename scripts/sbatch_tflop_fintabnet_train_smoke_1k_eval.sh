#!/bin/bash
#SBATCH --job-name=tflop_ftn1k_eval
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

SUBSET=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_smoke_1k_eval_subset
TRAIN_ROOT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_smoke_1k_training/tflop_fintabnet_train_smoke_1k/from_public_checkpoint_1000steps
FT_MODEL=${TRAIN_ROOT}/epoch_1_step_1000
EXP_CONFIG=${TRAIN_ROOT}/config.yaml
MODEL_CONFIG=${FT_MODEL}/config_infer_float16.json

BASE_MODEL=/cluster/home/trinhwin/vt2/docling/models/tflop
BASE_OUT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_smoke_1k_eval_baseline_public
FT_OUT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_smoke_1k_eval_epoch1_step1000
SUMMARY_OUT=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_smoke_1k_eval_summary.json

mkdir -p "$BASE_OUT" "$FT_OUT"

$PY_BUILD - <<'PY'
import json
from pathlib import Path
cfg = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_smoke_1k_training/tflop_fintabnet_train_smoke_1k/from_public_checkpoint_1000steps/epoch_1_step_1000/config.json")
out = cfg.with_name("config_infer_float16.json")
data = json.loads(cfg.read_text(encoding="utf-8"))
data["torch_dtype"] = "float16"
out.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"wrote {out}")
PY

run_eval () {
  local model_dir="$1"
  local out_dir="$2"
  local tokenizer_dir="$3"

  rm -f "$out_dir"/full_model_inference*.json \
        "$out_dir"/ted_score_output*.json \
        "$out_dir"/*.summary.json

  $PY_TFLOP $TFLOP_REPO/test.py \
    --tokenizer_name_or_path "$tokenizer_dir" \
    --model_name_or_path "$model_dir" \
    --exp_config_path "$EXP_CONFIG" \
    --model_config_path "$MODEL_CONFIG" \
    --aux_json_path "$SUBSET/aux.json" \
    --aux_img_path "$SUBSET/images" \
    --aux_rec_pkl_path "$SUBSET/aux_rec.pkl" \
    --batch_size 1 \
    --save_dir "$out_dir" \
    --use_validation

  $PY_TFLOP $TFLOP_REPO/evaluate_ted.py \
    --model_inference_pathdir "$out_dir" \
    --output_savepath "$out_dir"

  $PY_BUILD scripts/rescore_tflop_fintabnet_canonicalized.py \
    --run-dir "$out_dir" \
    --output-name ted_score_output_ftn_canonicalized.json \
    --num-processes 8
}

run_eval "$BASE_MODEL" "$BASE_OUT" "$BASE_MODEL"
run_eval "$FT_MODEL" "$FT_OUT" "$FT_MODEL"

$PY_BUILD - <<'PY'
import json
from pathlib import Path

runs = {
    "baseline_public_checkpoint": Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_smoke_1k_eval_baseline_public"),
    "finetuned_1k_steps_epoch1_step1000": Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_smoke_1k_eval_epoch1_step1000"),
}

def summarize_official(path: Path) -> dict:
    rows = json.loads((path / "ted_score_output.json").read_text(encoding="utf-8"))
    return {
        "samples": len(rows),
        "teds_s": sum(row[-2] for row in rows) / len(rows),
        "teds": sum(row[-1] for row in rows) / len(rows),
        "teds_s_1": sum(row[-2] == 1.0 for row in rows),
        "teds_1": sum(row[-1] == 1.0 for row in rows),
    }

summary = {
    name: {
        "official_tflop_eval": summarize_official(path),
        "ftn_section_canonicalized": json.loads(
            (path / "ted_score_output_ftn_canonicalized.summary.json").read_text(encoding="utf-8")
        ),
    }
    for name, path in runs.items()
}
out = Path("/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_train_smoke_1k_eval_summary.json")
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
