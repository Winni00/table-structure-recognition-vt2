# Data access and storage

## What is not stored in Git

The following material is deliberately excluded from the repository:

- PubTabNet and FinTabNet images, PDFs, and annotations
- the Antibiotic Paper Collection
- the Preliminary and Synchronised Fine-Tuning Datasets
- publisher XML and HTML references
- generated OCR records and pseudo-labels
- model checkpoints
- complete inference outputs, visualisations, and Slurm logs
- reports, feedback PDFs, and presentations

These files are too large for Git and some are access-restricted. Git LFS is
not appropriate for the project archive, which is hundreds of gigabytes and
contains many generated files.

## Recommended storage

Store the target-domain source data and final archival artefacts in a
restricted ZHAW OneDrive or SharePoint location. Access should be granted by
the data owner. Do not place a private share URL, session token, or personal
OneDrive path in this repository.

At the time of the project, the source material was supplied through a
supervisor-managed ZHAW SharePoint/OneDrive hierarchy under
`Student_Projects / trinhwin / Data`. This is a provenance note, not a stable
handover address. A successor must be granted access by the institutional data
owner to the final shared project archive. See [`HANDOVER.md`](HANDOVER.md) for
the required contents and access procedure.

A versioned archive can use this layout:

```text
table-recognition-data/
|-- README.md
|-- manifests/
|   |-- source_files.sha256
|   |-- prepared_files.sha256
|   `-- dataset_versions.csv
|-- public/
|   |-- pubtabnet/
|   `-- fintabnet_v1.0.0/
|-- restricted/
|   |-- antibiotic_paper_collection/
|   |-- preliminary_finetuning/
|   `-- synchronised_finetuning/
|-- checkpoints/
`-- archived_results/
```

Keep public datasets in local or cluster storage and recreate them from the
official download source when possible. The institutional archive only needs
to retain material that cannot be reconstructed reliably, such as the exact
target-domain release, pseudo-label manifests, selected checkpoints, and
final aggregate result tables.

## Expected local project layout

The historical scripts use several absolute paths. For a new installation,
map the following logical locations to your storage paths:

```text
data/
|-- pubtabnet/
|-- fintabnet/
|-- antibiotic_paper_collection/
|-- preliminary_finetuning/
`-- synchronised_finetuning/

models/
|-- tflop_public/
|-- tableformer/
`-- project_checkpoints/

results/
|-- tableformer/
|-- tflop_pubtabnet/
|-- tflop_fintabnet/
|-- antibiotic_paper_collection/
`-- target_domain_finetuning/
```

The directories themselves may exist in the checkout, but their contents are
ignored by Git.

## Dataset identities used in the report

### Antibiotic Paper Collection

- Development and evaluation collection, not an independent held-out test set
- Original crop version: 696 tables
- Updated crop version: 695 evaluable tables
- Used for crop, orientation, text-source, checkpoint, and qualitative analyses

### Preliminary Fine-Tuning Dataset

- Initial candidates: 12,934 train / 1,438 validation
- Accepted after pseudo-label quality filtering: 3,522 train / 417 validation
- Paper-level train/validation separation

### Synchronised Fine-Tuning Dataset

- Initial candidates: 13,254 train / 1,474 validation
- Accepted after pseudo-label quality filtering: 4,233 train / 505 validation
- Evaluated separately from the preliminary preparation

## Integrity manifests

For every shared dataset release, create a manifest without committing the
restricted filenames if they reveal sensitive information:

```bash
find /path/to/release -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > source_files.sha256
```

Record at least the release date, owner, number of papers, number of tables,
split method, accepted pseudo-label counts, and checksum-file location.

The template [`external-artifacts-manifest.example.tsv`](external-artifacts-manifest.example.tsv)
can be copied into the external archive and completed during handover.
