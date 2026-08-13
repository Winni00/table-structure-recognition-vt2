# Project handover

This document separates what is preserved in GitHub from the restricted and
large artefacts that must be handed over through institutional storage.

## 1. What GitHub contains

The private repository contains the curated experiment code, Slurm scripts,
configuration templates, dependency revisions, evaluation patches, compact
summary metrics, and reproduction documentation. It intentionally excludes
datasets, checkpoints, complete predictions, reports, presentations, and
temporary development outputs.

Start with these files:

1. [`README.md`](../README.md) for the project overview and setup.
2. [`DATA.md`](DATA.md) for dataset identities and the expected directory
   layout.
3. [`EXPERIMENTS.md`](EXPERIMENTS.md) for the mapping from experiments to
   scripts and outputs.
4. [`REPRODUCTION.md`](REPRODUCTION.md) for the ordered commands.
5. [`THIRD_PARTY.md`](THIRD_PARTY.md) for pinned external repositories and
   the TFLOP patch.

## 2. External handover archive

The following material is required for exact target-domain reproduction and
must be archived outside GitHub:

### Restricted datasets

- **Antibiotic Paper Collection**
  - permitted source PDFs
  - publisher XML and generated table-specific HTML
  - original 696-table crop version
  - updated 695-table crop version and crop-to-page metadata
  - orientation labels and the rotation-candidate list
- **Preliminary Fine-Tuning Dataset**
  - source/prepared table images and references
  - paper-level train/validation split manifests
  - generated PSENet--MASTER records and accepted pseudo-labels
  - 3,522 accepted training and 417 validation tables
- **Synchronised Fine-Tuning Dataset**
  - source/prepared table images and references
  - paper-level train/validation split manifests
  - generated PSENet--MASTER records and accepted pseudo-labels
  - 4,233 accepted training and 505 validation tables

### Selected checkpoints

- FinTabNet: FTN1K, FTN10K, FTN50K, and FTN100K
- mixed FinTabNet--PubTabNet 10K checkpoint
- Preliminary Fine-Tuning Dataset checkpoints: target only, 75:25 replay,
  and 50:50 replay
- Synchronised Fine-Tuning Dataset checkpoints: target only, 75:25 replay,
  and 50:50 replay

The public TFLOP and TableFormer weights do not need to be duplicated if their
official source, revision, and checksum are recorded. Locally trained weights
must be retained because they cannot be reconstructed without rerunning the
training jobs.

### Final result bundles

Retain the predictions and aggregate reports used for the final tables in the
report. Include the run configuration, data manifest, checkpoint checksum,
evaluator setting, and reference-canonicalisation setting. Scratch runs,
duplicate caches, and failed-job logs are optional and need not be handed over.

The suggested archive layout is:

```text
table-recognition-handover/
|-- README.md
|-- ACCESS.md
|-- manifests/
|-- restricted/
|   |-- antibiotic_paper_collection/
|   |-- preliminary_finetuning/
|   `-- synchronised_finetuning/
|-- checkpoints/
|   |-- fintabnet/
|   `-- target_domain/
`-- final_results/
```

`ACCESS.md` should name the institutional owner, access-request procedure,
archive creation date, and dataset version. It must not contain passwords,
session cookies, personal access tokens, or private browser share URLs.

## 3. Storage and access

Use a permission-controlled **ZHAW OneDrive or SharePoint project folder** as
the authoritative handover location. At the time of this project, source data
was supplied through a supervisor-managed ZHAW SharePoint/OneDrive hierarchy
under `Student_Projects / trinhwin / Data`. This records provenance but is not
a durable access method for a successor.

Before closing the project:

1. Copy the required artefacts into a stable ZHAW project folder owned by the
   responsible group rather than by a departing student's personal account.
2. Ask the data owner to grant the successor read access and the project owner
   administrative access.
3. Record the folder name, institutional owner, and access-request procedure in
   the archive's `ACCESS.md` and in the project handover record.
4. Test access with a second authorised account.

**Dropbox is not recommended.** The source material is institutionally
provided and may be access-restricted. Use Dropbox only if the data owner and
ZHAW explicitly approve that storage location. A public link must never be
used for restricted publisher material or checkpoints derived from it.

## 4. Integrity and installation

Create a SHA-256 manifest for the complete external archive:

```bash
python3 tools/build_data_manifest.py /path/to/table-recognition-handover \
  --output /path/to/table-recognition-handover/manifests/archive-files.tsv
```

After receiving access, a successor should:

1. Clone the private GitHub repository.
2. Download or synchronise the external archive to the cluster.
3. Recreate the logical directory layout from [`DATA.md`](DATA.md).
4. Copy `config/paths.example.sh` to `config/paths.local.sh` and configure the
   local paths.
5. Regenerate the manifest and compare it with the archived copy.
6. Run `python3 tools/check_repository.py` and then the public PubTabNet smoke
   reproduction before using restricted data.

## 5. Final handover checklist

- [ ] GitHub access granted to the successor and responsible supervisor
- [ ] stable institutional archive created
- [ ] institutional data owner and access procedure recorded
- [ ] three target-domain dataset versions archived with split manifests
- [ ] accepted pseudo-label bundles archived
- [ ] selected locally trained checkpoints archived
- [ ] final result bundles and evaluator settings archived
- [ ] SHA-256 manifest generated and verified from a second account
- [ ] clean clone and public smoke reproduction tested
- [ ] no presentation, report feedback, credentials, or restricted data in Git

