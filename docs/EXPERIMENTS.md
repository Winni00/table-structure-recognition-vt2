# Experiment map

This document maps the main project questions to the script families that
prepare data, launch jobs, merge outputs, and report metrics. Many scripts are
historical snapshots with cluster-specific paths. Inspect every Slurm file
before submission.

## 1. TableFormer reproduction

Purpose: reproduce PubTabNet and FinTabNet results with the Docling
TableFormer inference implementation and a custom input adapter.

Key entry points:

- `tableformer_pubtabnet_repro.py`
- `tableformer_pubtabnet_repro_bbox_experiments.py`
- `tableformer_fintabnet_kaggle_repro.py`
- `scripts/sbatch_tableformer_pubtabnet_val_9115.sh`
- `scripts/sbatch_tableformer_pubtabnet_val_72dpi_existingpng_size_1x1_20x10.sh`
- `scripts/sbatch_tableformer_pubtabnet_hf_reconstructed_full.sh`
- `scripts/sbatch_tableformer_fintabnet_val_72dpi_excl_problem_pages.sh`
- `scripts/rescore_tableformer_pubtabnet_run.py`

## 2. TFLOP PubTabNet reproduction

Purpose: verify the public checkpoint and official evaluation path on
PubTabNet validation and test data.

Key entry points:

- `build_tflop_pubtabnet_annotation_aux.py`
- `build_tflop_pubtabnet_val_aux_json.py`
- `build_tflop_ocr_inputs.py`
- `tflop_inference.py`
- `scripts/sbatch_tflop_pubtabnet_test_9064.sh`
- `scripts/sbatch_tflop_pubtabnet_test_recreated_ocr.sh`
- `scripts/sbatch_tflop_pubtabnet_val_pse_matched_filtered_8958.sh`
- `scripts/sbatch_rescore_tflop_pubtabnet_test_bundle_current_eval.sh`

## 3. FinTabNet controlled inputs

Purpose: compare annotation-cell boxes and text with PSENet text regions,
MASTER text, and spatially matched annotation text.

Key entry points:

- `build_tflop_fintabnet_annotation_aux.py`
- `scripts/prepare_tflop_fintabnet_ocr_style_inputs.py`
- `scripts/visualize_fintabnet_adapter_steps.py`
- `scripts/sbatch_tflop_fintabnet_full_annotation_excl_problem_pages.sh`
- `scripts/sbatch_tflop_fintabnet_ocr_style_full_ocr_array.sh`
- `scripts/sbatch_tflop_fintabnet_pse_matched_annotation_text_inference_array.sh`
- `scripts/sbatch_tflop_fintabnet_pse_matched_annotation_text_teds_array.sh`
- `scripts/sbatch_tflop_fintabnet_canonicalized_teds.sh`

## 4. FinTabNet fine-tuning

Purpose: adapt the public TFLOP checkpoint to FinTabNet and evaluate
cross-domain retention.

Key entry points:

- `scripts/build_ftn_trainval_tflop_dataset.py`
- `scripts/train_tflop_single_gpu.py`
- `scripts/train_tflop_single_gpu_train_only.py`
- `scripts/sbatch_tflop_fintabnet_train_smoke_1k_eval.sh`
- `scripts/sbatch_tflop_fintabnet_train_10k.sh`
- `scripts/sbatch_tflop_fintabnet_full_annotation_ftn50k_inference_array.sh`
- `scripts/sbatch_tflop_fintabnet_full_annotation_ftn100k_teds_array.sh`
- `scripts/sbatch_tflop_mixed_ftn16k_ptn4k_train_10k_lr1e5.sh`

FTN1K was a separate smoke test. FTN10K, FTN50K, and FTN100K were sequential
training stages with cumulative update counts of 10,000, 60,000, and 160,000.

## 5. Antibiotic Paper Collection pipeline

Purpose: evaluate crop preparation, orientation, OCR/PDF text sources, and
the public or fine-tuned TFLOP checkpoints on the target-domain collection.

Key entry points:

- `scripts/prepare_tflop_paper_collection_ocr_style_inputs.py`
- `scripts/prepare_paper_collection_rotation_candidates.py`
- `scripts/evaluate_paper_collection_orientation_detector.py`
- `scripts/sbatch_tflop_paper_collection_ocr_style_ocr_array.sh`
- `scripts/sbatch_tflop_paper_collection_ocr_style_inference_array.sh`
- `scripts/sbatch_tflop_paper_collection_updated_crops_ocr_array.sh`
- `scripts/sbatch_tflop_paper_collection_updated_crops_inference_array.sh`
- `scripts/report_paper_collection_ocr_style_results.py`
- `scripts/rescore_paper_collection_gt_canonicalized.py`

The `updated_crops` filenames refer to the updated target-domain crop
preparation and are retained to match completed job logs.

## 6. Text-source comparison

Purpose: keep PSENet regions and TFLOP fixed while comparing MASTER,
PyMuPDF, PARSeq, and TrOCR.

Key entry points:

- `scripts/build_paper_collection_pymupdf_text_from_run.py`
- `scripts/build_paper_collection_trocr_text_from_run.py`
- `scripts/compare_master_pymupdf_to_gt.py`
- `scripts/compare_master_vs_pymupdf_visuals.py`
- `scripts/generate_pymupdf_verification_gallery.py`
- `scripts/sbatch_tflop_paper_collection_bestrot_compare_master_pymupdf.sh`
- `scripts/sbatch_tflop_paper_collection_bestrot_parseq_inference_array.sh`
- `scripts/sbatch_tflop_paper_collection_bestrot_trocr_inference_array.sh`
- `scripts/summarize_master_pymupdf_text_experiment.py`

The score-guided orientation used by this comparison is diagnostic and uses
reference TEDS. It is not the deployable orientation pipeline.

## 7. Target-domain pseudo-labels and fine-tuning

Purpose: align ordered PSENet--MASTER text regions with publisher-derived
HTML cells, retain high-confidence tables, and compare target-only training
with PubTabNet replay.

Key entry points:

- `scripts/build_target_domain_tflop_training_dataset.py`
- `scripts/sbatch_prepare_target_domain_train_ocr.sh`
- `scripts/sbatch_merge_target_domain_train_ocr.sh`
- `scripts/sbatch_build_target_domain_replay_smoke_datasets.sh`
- `scripts/sbatch_train_synchronised_target_replay_smoke_array.sh`
- `scripts/sbatch_eval_synchronised_target_replay_smoke_array.sh`
- `scripts/report_synchronised_target_replay_smoke_results.py`

## Shared evaluation utilities

- `evaluate_table_benchmark.py`
- `scripts/evaluate_ted_shard.py`
- scripts named `*_merge_*`, `*_teds_*`, and `report_*`

Always use the report/merge script paired with the input and checkpoint
condition. Several experiments intentionally used different reference
canonicalisation settings and should not be combined silently.

