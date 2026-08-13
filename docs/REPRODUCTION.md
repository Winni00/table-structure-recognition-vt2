# Reproduction guide

## Scope

The project consists of several linked experiments rather than one monolithic
pipeline command. This guide gives the safe order for reproducing them. Use
the exact script list in [Experiment map](EXPERIMENTS.md) for each condition.

## Prerequisites

- Linux compute environment with Slurm
- NVIDIA GPU for OCR, TFLOP inference, and training
- Python 3.9 environment for the public TFLOP implementation
- separate analysis environment for current PyMuPDF-based diagnostics
- authorised dataset access and sufficient scratch storage
- third-party repositories at the commits in [Third-party software](THIRD_PARTY.md)

## 1. Configure paths

Copy the example and edit it outside Git if it contains private locations:

```bash
cp config/paths.example.sh config/paths.local.sh
chmod 600 config/paths.local.sh
```

Historical scripts currently contain absolute paths and do not all source
this file automatically. Update the relevant Slurm entry point before use.
Run `rg '/cluster/home|/Users/' scripts` to find path assumptions.

## 2. Prepare public benchmarks

Download PubTabNet and FinTabNet v1.0.0 from their official distribution
channels. Preserve the upstream split files. For FinTabNet validation, the
project excluded 34 table samples on 28 PDF pages whose coordinates did not
align reliably with rendered pages, leaving 10,622 annotation-input samples.
Two additional samples could not be processed in OCR-style conditions,
leaving 10,620.

Run a small conversion and render visual overlays before preparing the full
dataset.

## 3. Reproduce public checkpoints

Run in this order:

1. TableFormer PubTabNet smoke subset.
2. TableFormer PubTabNet full validation variants.
3. TFLOP PubTabNet validation smoke subset.
4. TFLOP PubTabNet validation or official 9,064-record test bundle.
5. Merge inference shards and compute TEDS-S/TEDS.

Do not compare scores produced with different HTML canonicalisation settings
as if they came from the same evaluator condition.

## 4. Build FinTabNet conditions

Create and visually inspect:

1. annotation boxes + annotation text;
2. PSENet boxes + MASTER text;
3. PSENet boxes + spatially matched annotation text.

The hybrid condition is not a recognition-only control: unmatched regions
are excluded and a full cell string may be assigned to a smaller PSENet
region.

## 5. Fine-tune on FinTabNet

Build the prepared train/validation records, then run the 1,000-update smoke
test. Only after validating loss curves, sample counts, and output HTML should
the sequential FTN10K, FTN50K, and FTN100K stages be launched. Evaluate
selected checkpoints on FinTabNet, PubTabNet, and the target-domain
collection to measure adaptation and retention.

## 6. Prepare target-domain inputs

The target-domain source provides table images and publisher XML from which
table-specific HTML references are prepared. It does not provide cell-level
image boxes. Generate PSENet regions, recognise them with MASTER, and convert
the result to TFLOP input.

For the fine-tuning datasets, align ordered regions to ordered non-empty HTML
cells and apply the pseudo-label quality thresholds encoded in
`scripts/build_target_domain_tflop_training_dataset.py`. Verify that train and
validation papers do not overlap.

## 7. Run the target-domain pipeline

1. Use the updated crop metadata.
2. Apply the image-only orientation detector before PSENet and MASTER.
3. Run TFLOP with the selected checkpoint.
4. Evaluate both TEDS-S and TEDS.
5. Label original-reference and canonicalised-reference scores explicitly.

Reference-guided orientation selection is only an upper-bound diagnostic and
must not be used as a production result.

## 8. Archive a run

For every final run, retain outside Git:

- submitted Slurm script and job IDs
- dependency commits and environment freeze
- data manifest/checksum and accepted sample count
- checkpoint identifier and checksum
- inference configuration
- merged per-table scores
- aggregate report
- representative error examples

Commit only small, non-sensitive summaries if redistribution is permitted.

