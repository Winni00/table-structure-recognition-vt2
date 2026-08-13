#!/usr/bin/env bash
#SBATCH --job-name=target_domainupd_eval
#SBATCH --partition=gpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail
ROOT=/cluster/home/trinhwin/vt2/docling
PY="$ROOT/.venv-tflop39/bin/python"
PY_SYS="$ROOT/.venv/bin/python"
TFLOP="$ROOT/repo/TFLOP"
EVAL="$ROOT/repo/TFLOP_clean/evaluate_ted.py"
EVAL_ROOT="$ROOT/results/tflop_synchronised_target_replay_smoke_1k_eval"
cd "$ROOT"

case "${SLURM_ARRAY_TASK_ID}" in
  0) VARIANT=public ;;
  1) VARIANT=target_domainupd_only ;;
  2) VARIANT=target_domainupd75_ptn25 ;;
  3) VARIANT=target_domainupd50_ptn50 ;;
  *) exit 2 ;;
esac

if [[ "$VARIANT" == public ]]; then
  MODEL_DIR="$ROOT/models/tflop"
  EXP_CONFIG="$ROOT/results/tflop_pubtabnet_official_repro/config.yaml"
else
  TRAIN_ROOT="$ROOT/results/tflop_synchronised_target_replay_smoke_1k/$VARIANT/tflop_$VARIANT/from_public_checkpoint_1000steps_lr1e5"
  EXP_CONFIG="$TRAIN_ROOT/config.yaml"
  MODEL_DIR=$($PY_SYS - "$TRAIN_ROOT" <<'PY'
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
candidates = []
for p in root.rglob("pytorch_model.bin"):
    m = re.search(r"step_(\d+)", p.parent.name)
    if m and (p.parent / "config.json").is_file():
        candidates.append((int(m.group(1)), p.parent))
if not candidates:
    raise SystemExit(f"No exported TFLOP checkpoint under {root}")
print(max(candidates)[1])
PY
  )
fi
MODEL_CONFIG="$MODEL_DIR/config_infer_float16.json"
$PY_SYS - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
cfg["torch_dtype"] = "float16"
(p / "config_infer_float16.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY

run_eval() {
  local DATASET="$1" INPUT="$2" IMAGES="$3"
  local OUT="$EVAL_ROOT/$VARIANT/$DATASET"
  mkdir -p "$OUT"
  ln -sfn "$INPUT/aux.json" "$OUT/aux.json"
  ln -sfn "$INPUT/aux_rec.pkl" "$OUT/aux_rec.pkl"

  "$PY" "$TFLOP/test.py" \
    --tokenizer_name_or_path "$MODEL_DIR" \
    --model_name_or_path "$MODEL_DIR" \
    --exp_config_path "$EXP_CONFIG" \
    --model_config_path "$MODEL_CONFIG" \
    --aux_json_path "$INPUT/aux.json" \
    --aux_img_path "$IMAGES" \
    --aux_rec_pkl_path "$INPUT/aux_rec.pkl" \
    --batch_size 1 \
    --save_dir "$OUT" \
    --use_validation

  if "$PY" "$EVAL" \
    --model_inference_pathdir "$OUT" \
    --output_savepath "$OUT"; then
    "$PY_SYS" scripts/summarize_ted_output.py \
      --input "$OUT/ted_score_output.json" \
      --output "$OUT/ted_score_output.summary.json"
  else
    echo "WARNING: official TEDS scoring failed for $VARIANT/$DATASET; retaining completed inference for robust or canonicalized rescoring." >&2
  fi
}

test -f "$ROOT/results/tflop_synchronised_target_val_eval_input/READY"
run_eval \
  target_domain_val \
  "$ROOT/results/tflop_synchronised_target_val_eval_input" \
  "$ROOT/results/tflop_synchronised_target_val_eval_input/images"
run_eval \
  ptn_val_pse \
  "$ROOT/results/tflop_pubtabnet_val_pse_matched_filtered_8958" \
  "$ROOT/data/TFLOP-dataset/images/validation"
run_eval \
  paper_test \
  "$ROOT/results/tflop_paper_collection_updated_crops_manualrot_master_public" \
  "$ROOT/results/tflop_paper_collection_updated_crops_manualrot_master_public/images"

"$PY" scripts/rescore_paper_collection_gt_canonicalized.py \
  --run-dir "$EVAL_ROOT/$VARIANT/paper_test" \
  --output-dir "$EVAL_ROOT/$VARIANT/paper_test/gt_canonicalized_rescore"

touch "$EVAL_ROOT/$VARIANT/COMPLETE"
