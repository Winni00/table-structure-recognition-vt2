# Environment notes

The project used separate environments for TFLOP and later analysis. These
versions document the working cluster state; they are not a universal lock
file for every script.

## TFLOP environment

- Python 3.9.19
- torch 2.0.1 (cluster installation used CUDA 11.7 build)
- torchvision 0.15.2 (CUDA 11.7 build)
- transformers 4.31.0
- pytorch-lightning 2.0.4
- timm 0.6.13
- numpy 1.26.4
- pandas 2.0.3
- Pillow 9.4.0
- lxml 4.9.3
- rapidfuzz 2.15.2
- apted 1.0.3
- Levenshtein 0.20.9

Start from the requirements supplied by the recorded TFLOP checkout. GPU
packages must match the CUDA version available on the execution node.

## Analysis environment

The current general analysis environment includes:

- Python 3.12.3
- PyMuPDF 1.27.2.3
- numpy 2.4.3
- pandas 3.0.1
- Pillow 12.1.1
- lxml 6.0.2
- RapidFuzz 3.14.3
- apted 1.0.3

Some older scripts may require versions closer to the TFLOP environment.
Create a dedicated environment for a specific workflow rather than upgrading
the historical environment in place.

