# Reproduction and Domain Adaptation of Table Recognition Pipelines

Research code and experiment documentation for the VT2 project on reproducing
and adapting **TableFormer** and **TFLOP** to scientific tables. The work
evaluates public benchmarks and a restricted collection of scientific tables,
with particular attention to input construction, OCR, orientation, HTML
representation, fine-tuning, and cross-domain retention.

> **Repository status:** this is a curated research artifact, not a packaged
> production library. It preserves the scripts and configurations used for the
> experiments. Historical Slurm scripts still contain cluster-specific paths;
> adapt them through `config/paths.example.sh` before running them elsewhere.

## Main findings

- The public TFLOP checkpoint reproduced the published PubTabNet test result:
  `0.9835` TEDS-S and `0.9662` TEDS versus `0.9838` and `0.9666` reported.
- FinTabNet-specific fine-tuning reached `0.9374` TEDS-S and `0.9293` TEDS
  under the unchanged evaluator. A separate HTML-canonicalised sensitivity
  analysis reached `0.9911` and `0.9829`.
- On the 695-table Antibiotic Paper Collection, updated crops plus automatic
  orientation reached `0.9225` TEDS-S and `0.8689` TEDS.
- MASTER was the strongest tested text source. Raw PyMuPDF recovered many
  reference tokens but added more unrelated tokens and had lower token F1.
- Target-domain fine-tuning improved the internal validation sets. PubTabNet
  replay reduced the loss of public-domain performance.

Scores using different reference canonicalisation rules are deliberately
reported separately. See [the experiment index](docs/EXPERIMENTS.md) for the
conditions behind every result family.

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/` | Experiment preparation, Slurm jobs, scoring, diagnostics, and reports |
| top-level `*.py` | Early TableFormer and TFLOP reproduction entry points |
| `config/paths.example.sh` | Portable path template for a local or cluster setup |
| `docs/EXPERIMENTS.md` | Mapping from experiment families to scripts and outputs |
| `docs/REPRODUCTION.md` | Ordered reproduction procedures |
| `docs/DATA.md` | Dataset identities, access policy, and expected layout |
| `docs/THIRD_PARTY.md` | Exact third-party revisions and TFLOP patch instructions |
| `environment/README.md` | Recorded software environments |
| `patches/` | Project-specific patch applied to the public TFLOP checkout |
| `tools/` | Repository and data-manifest validation utilities |

## Data policy

Large data, model checkpoints, complete experiment outputs, reports,
presentations, and feedback files are intentionally excluded from Git.

- **Public benchmarks:** PubTabNet and FinTabNet must be obtained from their
  original sources and remain subject to their respective licences.
- **Restricted project data:** the Antibiotic Paper Collection and the two
  fine-tuning datasets contain internally supplied publisher-derived material
  and are not redistributed here.
- **Recommended archive:** store restricted data and large derived artifacts in
  a permission-controlled ZHAW OneDrive/SharePoint folder. Add a file manifest,
  SHA-256 checksums, dataset version, creation date, and access contact. Do not
  use a public GitHub release or Git LFS for these files.

The exact dataset counts, identities, and suggested archive structure are in
[`docs/DATA.md`](docs/DATA.md). A manifest can be generated with:

```bash
python3 tools/build_data_manifest.py /path/to/dataset \
  --output /path/to/archive/dataset-manifest.tsv
```

## Setup

1. Clone this repository and the pinned third-party repositories.

   ```bash
   git clone https://github.com/winni00/table-structure-recognition-vt2.git
   cd table-structure-recognition-vt2
   mkdir -p external

   git clone https://github.com/UpstageAI/TFLOP external/TFLOP
   git -C external/TFLOP checkout d59ddf58a10fbaa22a7088b8f688bc6e694ae703
   git -C external/TFLOP am \
     ../../patches/tflop-evaluation-and-inference-diagnostics.patch
   ```

   The Docling revisions and all commands are listed in
   [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).

2. Recreate the recorded Python environments from
   [`environment/README.md`](environment/README.md). The original work used a
   Python 3.9 CUDA environment for TFLOP and a separate modern analysis
   environment. A single universal lock file would not accurately represent
   these two stacks.

3. Configure data and output paths.

   ```bash
   cp config/paths.example.sh config/paths.local.sh
   $EDITOR config/paths.local.sh
   source config/paths.local.sh
   ```

4. Follow [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md). Start with the public
   PubTabNet reproduction before running FinTabNet or restricted target-domain
   experiments.

## Reproduction scope

The public benchmark experiments can be reconstructed from public source data,
the pinned dependencies, and this repository. Exact target-domain reproduction
additionally requires authorised access to the restricted data archive and the
recorded derived OCR/pseudo-label bundles. Scripts are preserved to make the
processing decisions auditable even where source data cannot be published.

Before committing changes, run:

```bash
python3 -m compileall -q .
find . -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
python3 tools/check_repository.py
```

## Publication and licensing

This repository should remain **private** until the data owner and ZHAW have
confirmed which project code and documentation may be published. Third-party
code, model weights, datasets, publisher files, reports, and presentations are
not covered by this repository and must not be redistributed. No open-source
licence is granted for the project code at this stage.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Please also
cite the original TableFormer, TFLOP, PubTabNet, FinTabNet, OCR, and Docling
publications relevant to the experiment being reproduced.
