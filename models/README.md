# Model directory

Model weights and checkpoints are intentionally ignored by Git. Record their
source, training stage, checksum, and corresponding data manifest outside the
repository. Public dependency versions are listed in
[`docs/THIRD_PARTY.md`](../docs/THIRD_PARTY.md).

The minimum set of locally trained checkpoints that must be retained is listed
in [`docs/HANDOVER.md`](../docs/HANDOVER.md). Store these files in the
permission-controlled institutional archive, not in GitHub or a public model
registry.
