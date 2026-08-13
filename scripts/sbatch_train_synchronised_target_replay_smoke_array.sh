#!/usr/bin/env bash
#SBATCH --job-name=target_domainupd_replay_1k
#SBATCH --partition=gpu
#SBATCH --account=cai_exp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.out
#SBATCH --error=/cluster/home/trinhwin/vt2/docling/logs/%A_%a_%x.err
#SBATCH --mail-user=winhon24@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail
ROOT=/cluster/home/trinhwin/vt2/docling
PY="$ROOT/.venv-tflop39/bin/python"
case "${SLURM_ARRAY_TASK_ID}" in
    0)
        VARIANT=target_domainupd_only
        DATA_DIR="$ROOT/results/tflop_target_domain_train_updated_final_pseudo_labels"
        ;;
    1)
        VARIANT=target_domainupd75_ptn25
        DATA_DIR="$ROOT/results/tflop_mixed_target_domainupd2k_ptn667"
        ;;
    2)
        VARIANT=target_domainupd50_ptn50
        DATA_DIR="$ROOT/results/tflop_mixed_target_domainupd2k_ptn2k"
        ;;
    *) exit 2 ;;
esac
OUT="$ROOT/results/tflop_synchronised_target_replay_smoke_1k/$VARIANT"
cd "$ROOT"
test -s "$DATA_DIR/meta_data/dataset_train.jsonl"
test -s "$DATA_DIR/meta_data/dataset_validation.jsonl"

"$PY" scripts/train_tflop_single_gpu_train_only.py \
    --exp_config "$ROOT/repo/TFLOP/config/exp_configs/general_exp.yaml" \
    --data_config "$DATA_DIR/data_config.yaml" \
    exp_name="tflop_$VARIANT" \
    exp_version=from_public_checkpoint_1000steps_lr1e5 \
    result_path="$OUT" \
    pretrained_tokenizer_name_or_path="$ROOT/models/tflop" \
    pretrained_model_name_or_path="$ROOT/models/tflop" \
    max_length=1376 \
    max_position_embeddings=1376 \
    bbox_token_cnt=640 \
    train_batch_size=2 \
    val_batch_size=2 \
    num_workers=8 \
    lr=0.00001 \
    max_steps=1000 \
    max_epochs=-1 \
    save_every_n_train_steps=1000 \
    val_check_interval=250 \
    check_val_every_n_epoch=1 \
    use_OTSL=True \
    use_imgRoiAlign=True \
    use_RowWise_contLearning=True \
    use_ColWise_contLearning=True \
    use_bbox_HiMulConET=True \
    use_ptr_decoder=True \
    empty_cell_ptr_loss_coeff=0.5 \
    non_empty_cell_ptr_loss_coeff=0.5 \
    strategy=auto \
    precision=16-mixed \
    num_sanity_val_steps=0
