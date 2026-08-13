# Results directory

Full outputs, visualisations, and per-table predictions are intentionally
ignored by Git. The compact [`summary_metrics.csv`](summary_metrics.csv) file
records the principal values reported in the project report together with
their evaluation conditions. Scores with different reference-processing
settings must not be compared as if they came from the same evaluator.

Final runs should be archived in restricted project storage with their
configuration, checkpoint checksum, data manifest, and aggregate metrics, as
described in [`docs/DATA.md`](../docs/DATA.md).
