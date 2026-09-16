# LIBERO-90 cross-task transfer

Evaluation of the existing LIBERO-10-trained 500-update model on all 90 LIBERO-90
tasks. The [fixed protocol](../../LIBERO90_TRANSFER.md) declares 810 episodes,
three layouts/task, original H=5/H=20 and trained H=20, with no new training.
Task instructions overlapping added OPSD training are excluded from the primary
comparison and retained in the complete-suite report.

- `evaluations/step_{0,500}/raw/`: episode JSONL and task manifests.
- `evaluations/step_{0,500}/videos/`: episode videos, local and excluded from Git.
- `aggregated/`: success, paired differences, task outcomes and validation.
- `figures/`: success comparison and task-level transfer plots.
- `provenance/`: actual task/state audit, frozen checkpoint identities, resource
  accounting and verification records.
- `FINDINGS.md`: generated from the completed evaluation, regardless of direction.

Reproduction commands are in the protocol. All weights come from existing
checkpoints; inference uses P=50, ten flow steps, and the official execution loop.
Public base-dataset task metadata is recorded only as contextual exposure
information; it cannot establish absence from historical pretraining.
