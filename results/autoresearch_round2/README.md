# Round 2: longer OPSD training, horizons and suite transfer

This archive follows the [predeclared protocol](../../AUTORESEARCH_ROUND2.md).
The immutable matrix is [autoresearch_round2.yaml](../../configs/autoresearch_round2.yaml),
with 1,310 evaluation episodes across LIBERO-10/spatial/object/goal. The primary
comparison is the preselected step-500 model versus the original model at H=20,
on 100 paired LIBERO-10 episodes excluded from our OPSD training and screen.

Code revision for training/evaluation: `7bc24971126ab5e050d7e85db442f6b661695295`.
The initial job is `60078880`: one H100, eight CPU cores, 64 GiB host memory,
two-hour upper bound. Exact submission, environment and parent checkpoint are in
[resource_plan.json](provenance/resource_plan.json). Final Slurm accounting is
recorded separately after the job exits.

The parent is the complete [100-update pilot](../opsd_h20_100/README.md), pinned
by checkpoint manifest digest
`a7d6b7cb8cf42e2285721c007515b4a6526e4bb8c1874028f1dae841b5238a8c`.
Its FP32 master weights, EMA and Adam moments/counters are restored. Updates
101–500 keep the algorithm and optimizer unchanged; all training stays at H=20.
The simulator starts new episodes at each continuation boundary. Frozen
original/100/300/500 parameter snapshots share one policy server for evaluation.

All conditions use P=50 and 10 flow steps. This is the explicitly authorized
sequence-length extension of the official P=10 checkpoint and an experimental
temporal OPSD adaptation with Gaussian velocity matching. It is not a faithful
implementation of the image-generation Flow-OPD algorithm.

The [split audit](provenance/split_audit.json) verifies actual initial-state hashes.
Screening uses LIBERO-10 states 20–24, transfer states 20–22, and confirmation
states 30–39. Training uses LIBERO-10 indices 10–19; the prior 100 updates actually
visited indices 10–11. These indices identify initial object layouts, not single
observations or optimizer steps. The official checkpoint's original training data
is separate from this added OPSD train/evaluation split.

## Files

- `evaluations/{screen,transfer,confirmation}/step_*/raw/`: episode JSONL and
  complete task manifests, including source/config/checkpoint/RNG identities.
- `evaluations/*/step_*/videos/`: a video for each measured episode (local, not in Git).
- `aggregated/`: condition, task and episode CSVs; paired changes, confidence
  intervals, exact McNemar tests and multiplicity corrections.
- `figures/`: horizon, training-step, suite, compute/accuracy and loss plots.
- `training.jsonl`, `rollouts.jsonl`: every added optimizer update and rollout identity.
- `provenance/`: plan, checks, checkpoint lineage, Slurm resources and data audits.
- `stages.jsonl`: exact commands and stage timings, in execution order.
- `FINDINGS.md`: generated only from the completed measurements.

Full checkpoints and replayable training images/actions live under the scratch
path recorded in `provenance/resource_plan.json`, with milestone manifests and
file hashes. Keep that directory when moving the run; Git does not contain weights.

## Reproduce

Install dependencies and obtain the official checkpoint using the project
[README](../../README.md). The exact CPU checks, split audit, one-GPU submission,
individual H=25 evaluation and checkpoint-loading commands are in
[TRAINING.md](../../TRAINING.md#round-2-continue-to-500-updates-and-test-transfer).
Use fresh result/checkpoint paths for a rerun.

To recreate all tables, plots and findings after the complete matrix finishes:

```bash
source scripts/env.sh
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" \
  -m frequency_vla.study_analysis --results-dir results/autoresearch_round2
```

Run from the repository root. No GPU is needed for aggregation. The analyzer
requires every planned condition, checks the actual state hashes, parameter
snapshot, one-server identity and unchanged inference settings, and refuses
incomplete or mismatched data. No screen outcome selects a checkpoint.
